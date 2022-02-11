from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='saved_data/epoch_30_iou_0.97',
        visualize=True
    )
    generator.load_model()
    generator.run()
