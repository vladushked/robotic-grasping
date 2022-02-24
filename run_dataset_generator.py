from inference.dataset_generator import DatasetGenerator

if __name__ == '__main__':
    generator = DatasetGenerator(
        saved_model_path='trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98',
        visualize=True,
        enable_arm=False,
        include_depth=True,
        include_rgb=True,
        # conveyor_speed=0.1,
    )
    generator.load_model()
    generator.run()
