import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def show_data(data: List[torch.tensor], variable_name: str="x") -> None:
    """
    Function to display 2D shapes (set of 2D points).
    :param data: Set of 2D points
    :return:
    """
    interval = [0, 1]
    fig = plt.figure(figsize=(20, 8))
    for i, element in enumerate(data[:10]):
        ax = fig.add_subplot(2, 5, i+1)
        ax.set_xlim(interval)
        ax.set_ylim(interval)
        ax.set_xlabel("{}1".format(variable_name))
        ax.set_ylabel("{}2".format(variable_name))
        ax.set_box_aspect(1)
        ax.scatter(element[:, 0], element[:, 1], c="black", s=10)
    pass


def show_image_data_flat(data: List[torch.tensor]) -> None:
    """
    Function to display 2D shapes (set of 2D points).
    :param data: Set of 2D points
    :return:
    """
    cols = rows = int((1.0 * len(data)) ** 0.5)
    fig = plt.figure(figsize=(2 * cols, 2 * rows))
    for i, element in enumerate(data[:cols*rows]):
        ax = fig.add_subplot(cols, rows, i+1)
        ax.set_box_aspect(1)
        ax.set_axis_off()
        element = element.reshape((28, 28))
        ax.imshow(element, cmap="gray")
    pass


def scatter_2d_data_and_subset(data: np.array, subset: np.array) -> None:
    """
    Visualize a set of 2D points in x domain and a subset, which is colored differently and labeled
    :param data: Set of 2D points
    :param subset: Subset of 2D points that will be labeled
    :return:
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_box_aspect(1)
    ax.scatter(data[:, 0], data[:, 1], c="black", s=10)
    ax.scatter(subset[:, 0], subset[:, 1], c="blue", s=100)
    for i, element in enumerate(subset):
        ax.annotate("x{}".format(i), element, xytext=element + np.ones(2) * 0.01, c="blue", fontsize=16)
    #plt.savefig("image_scatter.png", bbox_inches='tight')
    pass


def show_densities(axis_scale: np.array, probabilities: np.array, variable_name: str) -> None:
    """
    Visualize 2D probability distributions on a regular grid.
    :param axis_scale: Axis positions for evaluated probability distribution.
    :param probabilities: Probabilities
    :param variable_name: Name of the space
    :return:
    """
    interval = [axis_scale.min(), axis_scale.max()]
    fig = plt.figure(figsize=(20, 8))
    for i, element in enumerate(probabilities[:10]):
        element_cpy = element.copy()
        ax = fig.add_subplot(2, 5, i+1)
        ax.set_xlim(interval)
        ax.set_ylim(interval)
        ax.set_xlabel("{}1".format(variable_name))
        ax.set_ylabel("{}2".format(variable_name))
        ax.set_box_aspect(1)
        element_cpy.flat[0] = 0.0
        element_cpy.flat[1] = 1.0
        ax.pcolormesh(axis_scale, axis_scale, element_cpy, shading="gouraud", cmap="gray")
    pass


def show_densities_grid(axis_scale: np.array, probabilities: np.array, variable_name: str) -> None:
    """
    Visualize 2D probability distributions on a regular grid.
    :param axis_scale: Axis positions for evaluated probability distribution.
    :param probabilities: Probabilities
    :param variable_name: Name of the space
    :return:
    """
    interval = [axis_scale.min(), axis_scale.max()]
    cols = rows = int((1.0 * len(probabilities)) ** 0.5)
    fig = plt.figure(figsize=(2 * cols, 2 * rows))
    for i, element in enumerate(probabilities[:cols * rows]):
        ax = fig.add_subplot(cols, rows, i + 1)
        ax.set_xlim(interval)
        ax.set_ylim(interval)
        ax.set_axis_off()
        ax.set_box_aspect(1)
        element.flat[0] = 0.0
        element.flat[1] = 1.0
        ax.pcolormesh(axis_scale, axis_scale, element, shading="gouraud", cmap="gray") #GnBu
    pass


"""
        # for saving figures
        plt.close(fig)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.set_box_aspect(1)
        #ax.pcolormesh(axis_scale, axis_scale, element, shading="gouraud", cmap="gray")
        ax.scatter(element[:, 0], element[:, 1], c="black", s=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.savefig("image_{}.png".format(i), bbox_inches='tight')
        continue
"""