from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98',
        visualize=True,
        enable_arm=True,
        include_depth=True,
        include_rgb=True,
        conveyor_speed=0.5,
        conveyor_direction="r_to_l",
        # conveyor_direction="l_to_r",
    )
    generator.load_model()
    generator.run()
