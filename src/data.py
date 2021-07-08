import numpy as np
import torch
from torchvision import datasets, transforms
from typing import List


def scale(sx: float, sy: float) -> np.array:
    """
    Returns scaling matrix.
    :param sx: Scaling factor for x axis
    :param sy: Scaling factor for y axis
    :return: Scaling matrix
    """
    return np.array([
        [sx, 0],
        [0, sy]
    ])


def rotate(alpha: float) -> np.array:
    """
    Returns rotation matrix.
    :param alpha: Rotation angle
    :return: Rotation matrix
    """
    return np.array([
        [np.cos(alpha),-np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])


def load_data() -> List[torch.tensor]:
    """
    Custom data loading function, that selects 10 images from the MNIST dataset, binarizes them and converts non-zero
    elements into a 2D data set of points.
    :return: data set
    """
    dataset = datasets.MNIST
    train_loader = torch.utils.data.DataLoader(
        dataset('data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=64, shuffle=False,
    )
    batch = None
    for batch in train_loader:
        break
    images = batch[0][[1, 3, 5, 7, 2, 0, 18, 15, 17, 4]].squeeze().numpy()
    #images = batch[0].squeeze().numpy()
    d = []
    for image in images:
        image_binarized = np.argwhere((1.0 * image > 0.1)).astype(np.int32) / image.shape[0]
        d.append(torch.from_numpy((image_binarized + np.array([-0.5, -0.5])) @ rotate(np.pi * 0.5) + np.array([0.5, 0.5])))
    return d


def load_image_data_flat() -> List[torch.tensor]:
    """
    Custom data loading function, that selects 10 images from the MNIST dataset and flattens them.
    :return: data set
    """
    dataset = datasets.MNIST
    train_loader = torch.utils.data.DataLoader(
        dataset('data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=64, shuffle=False,
    )
    batch = None
    for batch in train_loader:
        break
    #images = batch[0][[1, 3, 5, 7, 2, 0, 18, 15, 17, 4]].squeeze().numpy()
    images = batch[0].squeeze().numpy()
    d = []
    for image in images:
        d.append(image.flatten())
    d = torch.from_numpy(np.asarray(d))
    return d