import torch
from torch.utils.data import TensorDataset
import glob


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

    # train = [None, None]
    # test = [None, None]

    # # train data
    # for i in range(6):
    #     train_images_path = data_path + "/train_images_" + str(i) + ".pt"
    #     train_target_path = data_path + "/train_target_" + str(i) + ".pt"
    #     train_images = torch.load(train_images_path).view(5000, -1)
    #     train_target = torch.load(train_target_path)
    #     if train[0] is None:
    #         train[0] = train_images
    #     else:
    #         train[0] = torch.cat((train[0], train_images), dim=0)
    #     if train[1] is None:
    #         train[1] = train_target
    #     else:
    #         train[1] = torch.cat((train[1], train_target), dim=0)
    # print(train[0].shape)
    # print(train[1].shape)
    
    # # test data
    # test_images_path = data_path + "/test_images" + ".pt"
    # test_target_path = data_path + "/test_target" + ".pt"
    # test_images = torch.load(test_images_path).view(5000, -1)
    # test_target = torch.load(test_target_path)
    # test[0] = test_images
    # test[1] = test_target
    # print(test[0].shape)
    # print(test[1].shape)

    # return train, test

    train_images_list = glob.glob(data_path + "/train_images_*.pt")
    train_images_list = sorted(train_images_list, key=lambda name: name[13:])
    # print(train_images_list)
    train_images_list = [torch.load(x).unsqueeze(1) for x in train_images_list]
    train_images = torch.cat(train_images_list, dim=0)
    # print('Train set:', train_images.shape)

    train_target_list = glob.glob(data_path + "/train_target_*.pt")
    train_target_list = sorted(train_target_list, key=lambda name: name[13:])
    # print(train_target_list)
    train_target_list = [torch.load(x) for x in train_target_list]
    train_target = torch.cat(train_target_list, dim=0)
    train = TensorDataset(train_images, train_target)

    test_images = torch.load(data_path + "/test_images.pt").unsqueeze(1)
    # print('Test set:', test_images.shape)
    test_target = torch.load(data_path + "/test_target.pt")
    test = TensorDataset(test_images, test_target)
    
    return train, test

if __name__ == "__main__":
    mnist()