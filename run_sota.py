from inference.sota_generator import SotaGenerator

import torch

import grasp_det_seg.models as models
from grasp_det_seg.algos.detection import PredictionGenerator, ProposalMatcher, DetectionLoss
from grasp_det_seg.algos.fpn import DetectionAlgoFPN, RPNAlgoFPN
from grasp_det_seg.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from grasp_det_seg.algos.semantic_seg import SemanticSegAlgo, SemanticSegLoss
from grasp_det_seg.config import load_config
from grasp_det_seg.data_OCID import iss_collate_fn, OCIDTestDataset, OCIDTestTransform
from grasp_det_seg.data_OCID.OCID_class_dict import colors_list, cls_list
from grasp_det_seg.data_OCID.sampler import DistributedARBatchSampler
from grasp_det_seg.models.det_seg import DetSegNet, NETWORK_INPUTS
from grasp_det_seg.modules.fpn import FPN, FPNBody
from grasp_det_seg.modules.heads import RPNHead, FPNROIHead, FPNSemanticHeadDeeplab
from grasp_det_seg.utils import logging
from grasp_det_seg.utils.meters import AverageMeter
from grasp_det_seg.utils.misc import config_to_string, scheduler_from_config, norm_act_from_config, freeze_params, \
    all_reduce_losses, NORM_LAYERS, OTHER_LAYERS
from grasp_det_seg.utils.parallel import DistributedDataParallel
from grasp_det_seg.utils.snapshot import resume_from_snapshot

def make_config(path):
    print("Loading configuration from %s", path)
    conf = load_config(path, path)
    return conf

def make_model(config):
    body_config = config["body"]
    fpn_config = config["fpn"]
    rpn_config = config["rpn"]
    roi_config = config["roi"]
    sem_config = config["sem"]
    general_config = config["general"]
    classes = {"total": int(general_config["num_things"]) + int(general_config["num_stuff"]), "stuff":
        int(general_config["num_stuff"]), "thing": int(general_config["num_things"]),
               "semantic": int(general_config["num_semantic"])}

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    print("Creating backbone model %s", body_config["body"])
    body_fn = models.__dict__["net_" + body_config["body"]]
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, **body_params)
    if body_config.get("weights"):
        body.load_state_dict(torch.load(body_config["weights"], map_location="cpu"))

    # Freeze parameters
    for n, m in body.named_modules():
        for mod_id in range(1, body_config.getint("num_frozen") + 1):
            if ("mod%d" % mod_id) in n:
                freeze_params(m)

    body_channels = body_config.getstruct("out_channels")

    # Create FPN
    fpn_inputs = fpn_config.getstruct("inputs")
    fpn = FPN([body_channels[inp] for inp in fpn_inputs],
              fpn_config.getint("out_channels"),
              fpn_config.getint("extra_scales"),
              norm_act_static,
              fpn_config["interpolation"])
    body = FPNBody(body, fpn, fpn_inputs)

    # Create RPN
    proposal_generator = ProposalGenerator(rpn_config.getfloat("nms_threshold"),
                                           rpn_config.getint("num_pre_nms_train"),
                                           rpn_config.getint("num_post_nms_train"),
                                           rpn_config.getint("num_pre_nms_val"),
                                           rpn_config.getint("num_post_nms_val"),
                                           rpn_config.getint("min_size"))
    anchor_matcher = AnchorMatcher(rpn_config.getint("num_samples"),
                                   rpn_config.getfloat("pos_ratio"),
                                   rpn_config.getfloat("pos_threshold"),
                                   rpn_config.getfloat("neg_threshold"),
                                   rpn_config.getfloat("void_threshold"))
    rpn_loss = RPNLoss(rpn_config.getfloat("sigma"))
    rpn_algo = RPNAlgoFPN(
        proposal_generator, anchor_matcher, rpn_loss,
        rpn_config.getint("anchor_scale"), rpn_config.getstruct("anchor_ratios"),
        fpn_config.getstruct("out_strides"), rpn_config.getint("fpn_min_level"), rpn_config.getint("fpn_levels"))
    rpn_head = RPNHead(
        fpn_config.getint("out_channels"), len(rpn_config.getstruct("anchor_ratios")), 1,
        rpn_config.getint("hidden_channels"), norm_act_dynamic)

    # Create detection network
    prediction_generator = PredictionGenerator(roi_config.getfloat("nms_threshold"),
                                               roi_config.getfloat("score_threshold"),
                                               roi_config.getint("max_predictions"))
    proposal_matcher = ProposalMatcher(classes,
                                       roi_config.getint("num_samples"),
                                       roi_config.getfloat("pos_ratio"),
                                       roi_config.getfloat("pos_threshold"),
                                       roi_config.getfloat("neg_threshold_hi"),
                                       roi_config.getfloat("neg_threshold_lo"),
                                       roi_config.getfloat("void_threshold"))
    roi_loss = DetectionLoss(roi_config.getfloat("sigma"))
    roi_size = roi_config.getstruct("roi_size")
    roi_algo = DetectionAlgoFPN(
        prediction_generator, proposal_matcher, roi_loss, classes, roi_config.getstruct("bbx_reg_weights"),
        roi_config.getint("fpn_canonical_scale"), roi_config.getint("fpn_canonical_level"), roi_size,
        roi_config.getint("fpn_min_level"), roi_config.getint("fpn_levels"))
    roi_head = FPNROIHead(fpn_config.getint("out_channels"), classes, roi_size, norm_act=norm_act_dynamic)

    # Create semantic segmentation network
    sem_loss = SemanticSegLoss(ohem=sem_config.getfloat("ohem"))
    sem_algo = SemanticSegAlgo(sem_loss, classes["semantic"])
    sem_head = FPNSemanticHeadDeeplab(fpn_config.getint("out_channels"),
                                      sem_config.getint("fpn_min_level"),
                                      sem_config.getint("fpn_levels"),
                                      classes["semantic"],
                                      pooling_size=sem_config.getstruct("pooling_size"),
                                      norm_act=norm_act_static)

    # Create final network
    return DetSegNet(body, rpn_head, roi_head, sem_head, rpn_algo, roi_algo, sem_algo, classes)

if __name__ == '__main__':
    

    config_path = 'trained-models/model_last.pth.tar'
    model_path = 'trained-models/model_last.pth.tar'

    # Load configuration
    config = make_config(config_path)

    # Create model
    model = make_model(config)

    print("Loading snapshot from %s", model_path)
    snapshot = resume_from_snapshot(model, model_path, ["body", "rpn_head", "roi_head", "sem_head"])

    generator = SotaGenerator(
        cam_id=830112070066,
        model=model,
        visualize=True,
        enable_arm=False,
        include_depth=True,
        include_rgb=False
    )
    generator.run()