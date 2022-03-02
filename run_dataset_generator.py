from inference.dataset_generator import DatasetGenerator

if __name__ == '__main__':
    generator = DatasetGenerator(
        saved_model_path='trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98',
        dataset_dir='/home/jetsonuser/data/leolab_grasping_dataset',
        material_dir='plastic/PET/heat_shrink_label',
    )
    generator.load_model()
    generator.run()
