from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97',
        visualize=True,
        enable_arm=False,
        include_depth=True,
        include_rgb=True
    )
    generator.load_model()
    generator.run()
