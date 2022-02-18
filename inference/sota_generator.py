import os
import time
import cv2
import scipy

import matplotlib.pyplot as plt
import numpy as np
import torch

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp, plot_results

from RAS_Com import RAS_Connect

from grasp_det_seg.utils.parallel import PackedSequence


class SotaGenerator:
    def __init__(self, model, cam_id, visualize=False, enable_arm=False, include_depth=True,
                 include_rgb=True):

        self.width = 640
        self.height = 480
        self.output_size = 200
        self.grip_height = 0.5

        self.enable_arm = enable_arm

        self.camera = RealSenseCamera(device_id=cam_id,
                                      width=self.width,
                                      height=self.height,
                                      fps=30)

        self.model = model
        self.device = None

        self.cam_data = CameraData(width=self.width,
                                   height=self.height,
                                   output_size=self.output_size,
                                   include_depth=include_depth,
                                   include_rgb=include_rgb)

        # Connect to camera
        self.camera.connect()

        # Load camera pose and depth scale (from running calibration)
        self.cam_pose = np.loadtxt('saved_data/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt(
            'saved_data/camera_depth_scale.txt', delimiter=' ')

        # homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        # self.grasp_request = os.path.join(homedir, "grasp_request.npy")
        # self.grasp_available = os.path.join(homedir, "grasp_available.npy")
        # self.grasp_pose = os.path.join(homedir, "grasp_pose.npy")

        if visualize:
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

        if self.enable_arm:
            self.s = RAS_Connect('/dev/ttyTHS0')

    def Rotate2D(self, pts, cnt, ang):
        ang = np.deg2rad(ang)
        return scipy.dot(pts - cnt, scipy.array([[scipy.cos(ang), scipy.sin(ang)], [-scipy.sin(ang),
                                                                                    scipy.cos(ang)]])) + cnt

    def show_prediction_image(self, img, raw_pred):

        num_classes_theta = 18
        # grasp candidate confidence threshold
        threshold = 0.5

        iou_seg_threshold = 100  # in px

        for (sem_pred, bbx_pred, cls_pred, obj_pred) in zip(raw_pred["sem_pred"], raw_pred["bbx_pred"], raw_pred["cls_pred"], raw_pred["obj_pred"]):

            sem_pred = np.asarray(
                sem_pred.detach().cpu().numpy(), dtype=np.uint8)
            print("sem_pred", sem_pred)
            if bbx_pred is None:
                continue
            print("bbx_pred", bbx_pred)
            print("cls_pred", cls_pred)
            print("obj_pred", obj_pred)

            img_best_boxes = np.copy(img)
            best_confidence = 0.
            r_bbox_best = None
            cls_labels = np.unique(sem_pred)

            for label in cls_labels:
                for bbx_pred_i, cls_pred_i, obj_pred_i in zip(bbx_pred, cls_pred, obj_pred):
                    if obj_pred_i.item() > threshold:

                        pt1 = (int(bbx_pred_i[0]), int(bbx_pred_i[1]))
                        pt2 = (int(bbx_pred_i[2]), int(bbx_pred_i[3]))
                        cls = cls_pred_i.item()
                        if cls > 17:
                            assert False

                        theta = ((180 / num_classes_theta) * cls) + 5
                        pts = scipy.array([[pt1[0], pt1[1]], [pt2[0], pt1[1]], [
                                        pt2[0], pt2[1]], [pt1[0], pt2[1]]])
                        cnt = scipy.array([(int(bbx_pred_i[0]) + int(bbx_pred_i[2])) / 2,
                                        (int(bbx_pred_i[1]) + int(bbx_pred_i[3])) / 2])
                        r_bbox_ = self.Rotate2D(pts, cnt, 90 - theta)
                        r_bbox_ = r_bbox_.astype('int16')

                        if (int(cnt[1]) >= self.width) or (int(cnt[0]) >= self.height):
                            continue

                        
                        if obj_pred_i.item() >= best_confidence:
                            best_confidence = obj_pred_i.item()
                            r_bbox_best = r_bbox_

                        cv2.line(img_best_boxes, tuple(r_bbox_[0]), tuple(
                            r_bbox_[1]), (255, 0, 0), 2)
                        cv2.line(img_best_boxes, tuple(r_bbox_[1]), tuple(
                            r_bbox_[2]), (0, 0, 255), 2)
                        cv2.line(img_best_boxes, tuple(r_bbox_[2]), tuple(
                            r_bbox_[3]), (255, 0, 0), 2)
                        cv2.line(img_best_boxes, tuple(r_bbox_[3]), tuple(
                            r_bbox_[0]), (0, 0, 255), 2)

            res = np.hstack((img, img_best_boxes))
            print("res.shape" ,res.shape)
            scale_percent = 75  # percent of original size
            width = int(res.shape[1] * scale_percent / 100)
            height = int(res.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(res, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("Result", resized)
            cv2.waitKey(0)

    def generate(self):
        # Get RGB-D image from camera
        image_bundle = self.camera.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)
        print(rgb_img.shape)
        rgb = rgb_img.transpose((1, 2, 0))
        print(rgb)
        cv2.imshow("rgb", rgb)
        # cv2.imshow("x", x)
        print(x[0].shape)

        # Predict the grasp pose using the saved model
        with torch.no_grad():
            xc = x[0].to(self.device)
            # Run network
            _, pred, conf = self.model(img=PackedSequence(
                xc), do_loss=False, do_prediction=True)
            # pred = self.model.predict(xc)
        
        self.show_prediction_image(rgb, pred)

        return None, None

        q_img, ang_img, width_img = post_process_output(
            pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasps = detect_grasps(q_img, ang_img, width_img,  no_grasps=10)

        if self.fig:
            # plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, grasp_q_img=q_img,
            #            grasp_angle_img=ang_img,
            #            no_grasps=10,
            #            grasp_width_img=width_img)

            plot_results(fig=self.fig,
                         rgb_img=self.cam_data.get_rgb(rgb, False),
                         depth_img=np.squeeze(self.cam_data.get_depth(depth)),
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         no_grasps=10,
                         grasp_width_img=width_img)

        if len(grasps) == 0:
            return None, None

        # Get grasp position from model output
        pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0],
                      grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale
        pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
                            pos_z / self.camera.intrinsics.fx)
        pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
                            pos_z / self.camera.intrinsics.fy)

        if pos_z == 0:
            return None, None

        target = np.asarray([pos_x, pos_y, pos_z])
        target.shape = (3, 1)
        #print('target: ', target)

        # Convert camera to robot coordinates
        camera2robot = self.cam_pose
        target_position = np.dot(
            camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:]
        target_position = target_position[0:3, 0]

        # Convert camera to robot angle
        angle = np.asarray([0, 0, grasps[0].angle])
        angle.shape = (3, 1)
        target_angle = np.dot(camera2robot[0:3, 0:3], angle)

        # Concatenate grasp pose with grasp angle
        grasp_pose = np.append(target_position, target_angle[2])

        # print('grasp_pose: ', grasp_pose)

        return grasp_pose, grasps[0].width

    def run(self):
        if self.enable_arm:
            while(True):
                print("Resetting position")
                self.s.grip(90)
                time.sleep(2)
                self.s.effectorMovement(0, 150, 300, 0)
                time.sleep(2)
                tool_position, grasp_width = self.generate()
                if tool_position is None:
                    continue

                z = tool_position[2] * 0.5
                if tool_position[2] > self.grip_height:
                    z = tool_position[2] - self.grip_height * 0.5
                print("___POSITION___: ", tool_position)
                print("___ANGLE___: ", tool_position[3] * 100)
                print("___Z___: ", z * 1000)
                print("___LENGTH___", grasp_width)
                print("___WIDTH___", grasp_width)
                # self.s.grip()
                self.s.effectorMovement(
                    tool_position[0] * 1000, tool_position[1] * 1000, z * 1000 + 50, - tool_position[3] * 100 * 0.5 * 0.62)
                # self.s.effectorMovement(0, 300, 300, tool_position[3] * 1000)
                time.sleep(2)
                self.s.effectorMovement(
                    tool_position[0] * 1000, tool_position[1] * 1000, z * 1000, - tool_position[3] * 100 * 0.5 * 0.62)
                time.sleep(2)
                self.s.grip(0)
                time.sleep(2)
                self.s.effectorMovement(
                    tool_position[0] * 1000, tool_position[1] * 1000, 300, - tool_position[3] * 100 * 0.5 * 0.62)
                time.sleep(2)
                self.s.effectorMovement(-400, 200, 300, 0)
                time.sleep(2)

        else:
            while(True):
                tool_position = self.generate()
                time.sleep(1)
