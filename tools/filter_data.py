from det3d.datasets.dataset_factory import get_dataset

if __name__ == "__main__":
    root = "/home/liyue/workspace/datasets/KITTI"
    source = root + "/kitti_infos_train.pkl"
    file = "/home/liyue/workspace/datasets/KITTI/filtered.pkl"
    
    pipeline = [
        {"type": "LoadPointCloudFromFile", "dataset": "KittiDataset",},
        {"type": "LoadPointCloudAnnotations", "with_bbox": True, "enable_difficulty_level": True},
    ]
    
    # get KittiDataset loaded points and annos.
    dataset = get_dataset("KITTI")(root, source, test_mode=True, pipeline=pipeline)
    dataset.dump_data_with_filter(file, ["Pedestrian", "Cyclist"])

