from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93',
        visualize=True
    )
    generator.load_model()
    generator.run()
