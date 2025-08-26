import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate
import BaseASD.DifferNet.config as c
from BaseASD.DifferNet.multi_transform_loader import ImageFolderMultiTransform


def get_random_transforms():
    augmentative_transforms = []
    if c.transf_rotations:
        augmentative_transforms += [transforms.RandomRotation(180)]
    if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
        augmentative_transforms += [transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
                                                           saturation=c.transf_saturation)]

    tfs = [transforms.Resize(c.img_size)] + augmentative_transforms + [transforms.ToTensor(),
                                                                       transforms.Normalize(c.norm_mean, c.norm_std)]

    transform_train = transforms.Compose(tfs)
    return transform_train


def get_fixed_transforms(degrees):
    cust_rot = lambda x: rotate(x, degrees, False, False, None)
    augmentative_transforms = [cust_rot]
    if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
        augmentative_transforms += [
            transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
                                   saturation=c.transf_saturation)]
    tfs = [transforms.Resize(c.img_size)] + augmentative_transforms + [transforms.ToTensor(),
                                                                       transforms.Normalize(c.norm_mean,
                                                                                            c.norm_std)]
    return transforms.Compose(tfs)


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]


def load_datasets(dataset_path, class_name):
    '''
    Expected folder/file format to find anomalies of class <class_name> from dataset location <dataset_path>:

    train data:

            dataset_path/class_name/train/good/any_filename.png
            dataset_path/class_name/train/good/another_filename.tif
            dataset_path/class_name/train/good/xyz.png
            [...]

    test data:

        'normal data' = non-anomalies

            dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
            dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
            dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
            dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
            dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
            [...]

        anomalies - assume there are anomaly classes 'crack' and 'curved'

            dataset_path/class_name/test/crack/dat_crack_damn.png
            dataset_path/class_name/test/crack/let_it_crack.png
            dataset_path/class_name/test/crack/writing_docs_is_fun.png
            [...]

            dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
            dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
            [...]
    '''

    def target_transform(target):
        return class_perm[target]

    # 获取训练数据集的路径
    data_dir_train = os.path.join(dataset_path, class_name, 'train')
    # 获取测试数据集的路径
    data_dir_test = os.path.join(dataset_path, class_name, 'test')

    # 获取测试数据集中的所有类别
    classes = os.listdir(data_dir_test)
    # 如果没有'good'子目录，则退出
    if 'good' not in classes:
        print('There should exist a subdirectory "good". Read the doc of this function for further information.')
        exit()
    # 对类别进行排序
    classes.sort()
    # 创建一个空列表，用于存储类别的索引
    class_perm = list()
    # 类别索引从1开始
    class_idx = 1
    # 遍历所有类别
    for cl in classes:
        # 如果类别是'good'，则索引为0
        if cl == 'good':
            class_perm.append(0)
        else:
            # 否则，索引为class_idx，并将class_idx加1
            class_perm.append(class_idx)
            class_idx += 1

    # 获取训练数据的随机变换
    transform_train = get_random_transforms()

    # 加载训练数据集
    trainset = ImageFolderMultiTransform(data_dir_train, transform=transform_train, n_transforms=c.n_transforms)
    # 加载测试数据集
    testset = ImageFolderMultiTransform(data_dir_test, transform=transform_train, target_transform=target_transform,
                                        n_transforms=c.n_transforms_test)
    # 返回训练数据集和测试数据集
    return trainset, testset


def load_eval_dataset(dataset_path, class_name):
    """

    """
    # data_dir_eval = os.path.join(dataset_path, class_name, 'eval')
    transform_train = get_random_transforms()

    testeval = ImageFolderMultiTransform(dataset_path, transform=transform_train, n_transforms=c.n_transforms_test)
    return testeval


def make_dataloaders(trainset, testset):
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                              drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.batch_size_test, shuffle=True,
                                             drop_last=False)
    return trainloader, testloader


def preprocess_batch(data):
    '''move data to device and reshape image'''
    inputs, labels = data
    inputs, labels = inputs.to(c.device), labels.to(c.device)
    inputs = inputs.view(-1, *inputs.shape[-3:])
    return inputs, labels

