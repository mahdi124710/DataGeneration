import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Union
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_time_series(x: np.array, open_price: float = 100) -> None:
    """

    :param x: The matrix of time series which is N by T
    :param open_price: The first price of time series, which is mutual for all
    :return:
    """
    plt.plot(np.cumsum(x, axis=1).T + open_price)
    plt.show()


def plot_correlation_matrices(corr_mx: np.array, *, axis: plt.Axes = None, **kwargs) -> Union[None, plt.Axes]:
    """
    This function will plot correlation matrices

    :param axis: The plt.Axis that you want to plot on. When this value is None the function will plot in regular way.
    But when the value is not None the corr_mx should have the shape (dim, dim) and it is plotted on the axis and the
    axis will be returned by the function
    :param corr_mx: A 2d or 3d numpy array of correlation matrices
    :param kwargs: You can define the min and max for the colorbar
    :return:
    """

    min_value, max_value = 0, 0
    if 'min' in kwargs.keys():
        min_value = kwargs['min']
    if 'max' in kwargs.keys():
        max_value = kwargs['max']

    if axis is not None:
        im = axis.imshow(corr_mx, origin='lower', vmin=0, vmax=1)
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axis.set_title('Correlation matrix')

        return axis

    if len(corr_mx.shape) == 2:
        corr_mx = corr_mx[None, :, :]

    n, dim1 = corr_mx.shape[0], int(math.sqrt(corr_mx.shape[0]))
    dim2 = dim1 if dim1 * dim1 == n else dim1 + 1
    dim1 = dim1 if dim1 * (dim1 + 1) >= n else dim1 + 1

    for i in range(n):
        plt.subplot(dim1, dim2, i + 1)
        plt.imshow(corr_mx[i], origin='lower',
                   vmin=np.amin(corr_mx[i]) if 'min' not in kwargs.keys() else min_value,
                   vmax=np.amax(corr_mx[i]) if 'max' not in kwargs.keys() else max_value)
        plt.colorbar()
    plt.show()
