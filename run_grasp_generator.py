from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94',
        visualize=True
    )
    generator.load_model()
    generator.run()
