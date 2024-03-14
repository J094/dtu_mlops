import torch
import os


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    data_path = "/home/jun/workspace/codes/dtu_mlops/data/corruptmnist"
    
    """Use os.walk to traverse the data path"""
    # for dirpath, dirnames, filenames in os.walk(data_path):
    #     # 输出当前遍历的路径
    #     print("路径：", dirpath)
    #     # 输出当前路径下的文件夹名
    #     print("文件夹：", dirnames)
    #     # 输出当前路径下的文件名
    #     print("文件名：", filenames)
    
    """Use os.listdir to traverse the data path"""
    """Use os.path to get file path"""
    """Use endswith to filter file suffix"""
    """Use startwith to filter file prefix"""
    """Use os.path.splitext to get suffix"""
    # for filename in os.listdir(data_path):
    #     print("文件名：", filename)
    #     filepath = os.path.join(data_path, filename)
    #     print("文件路径：", filepath)
    #     if filename.endswith(".pt"):
    #         name, suffix = os.path.splitext(filename)
    #         print("文件后缀：", suffix)
    #         print(name[-1])
    #     if filename.startswith("test"):
    #         print("文件前缀：test")

    train = {"images": None, "target": None}
    test = {"images": None, "target": None}
    for filename in os.listdir(data_path):
        if filename.startswith("train_images"):
            train_images_tmp = torch.load(os.path.join(data_path, filename))
            train_images_tmp = train_images_tmp.view(train_images_tmp.shape[0], -1)
            if train["images"] is None:
                train["images"] = train_images_tmp
        if filename.startswith("train_target"):
            train_target_tmp = torch.load(os.path.join(data_path, filename))
            if train["target"] is None:
                train["target"] = train_target_tmp
        if filename.startswith("test_images"):
            test_images_tmp = torch.load(os.path.join(data_path, filename))
            test_images_tmp = test_images_tmp.view(test_images_tmp.shape[0], -1)
            if test["images"] is None:
                test["images"] = test_images_tmp
        if filename.startswith("test_target"):
            test_target_tmp = torch.load(os.path.join(data_path, filename))
            if test["target"] is None:
                test["target"] = test_target_tmp
    print(train["images"].shape)
    print(train["target"].shape)
    print(test["images"].shape)
    print(test["target"].shape)
    return train, test


if __name__ == "__main__":
    mnist()