import numpy as np
from typing import Tuple, Union, List


def __sample_from_distribution(distribution: str, size: Union[int, Tuple]) -> np.array:
    """
    This function would return a numpy array of the specified size, and each element is sampled from a distribution
    specified in 'distribution' argument.

    :param distribution: A string of the name of distribution which can take value in ['normal', 'normal_2', 'laplace',
    't-student']
    :param size: An int or tuple which shows the size of the output
    :return: A numpy array with each element drawn independently from distribution and with size=size
    """

    sampled_data = np.array([])
    if distribution == 'normal':
        sampled_data = np.random.normal(0, 1, size)
    elif distribution == 'normal_2':
        sampled_data = np.random.normal(0, 2, size)
    elif distribution == 'laplace':
        sampled_data = np.random.laplace(0, 1 / np.sqrt(2), size=size)
    elif distribution == 't-student':
        sampled_data = np.random.standard_t(3, size=size) / np.sqrt(3)
    return sampled_data


def generate_correlated_random_walks(*,
                                     n_series: int,
                                     t_points: int,
                                     k_corr_clusters: int,
                                     d_dist_clusters: int,
                                     beta: float,
                                     dists: List[str],
                                     rho_main: float) -> np.array:
    """
    Reference: `Donnat, P., Marti, G. and Very, P., 2016. Toward a generic representation of random
    variables for machine learning. Pattern Recognition Letters, 70, pp.24-31.
    <https://www.sciencedirect.com/science/article/pii/S0167865515003906>`_

    :param n_series: Number of time series to generate
    :param t_points: Number of samples in each time series
    :param k_corr_clusters: Number of correlated clusters
    :param d_dist_clusters: Number of distribution clusters in each correlated cluster
    :param beta: The amount of correlation in each correlation cluster
    :param dists: A list of names of distributions to use for distribution clusters. The first element is used to
    generate correlations and the others are used for distribution clusters. Names should take value in ['normal',
     'normal_2', 'laplace', 't-student']
    :param rho_main: The amount of the main time series underlying all time series
    :return:
    """

    if n_series % (k_corr_clusters * d_dist_clusters) != 0:
        raise Exception('n_series should be divisible by (k_corr_clusters * d_dist_clusters)')

    y = __sample_from_distribution(dists[0], size=(k_corr_clusters, t_points))
    x = rho_main * np.repeat(np.random.normal(0, 1, t_points)[None, :], repeats=n_series, axis=0)

    y_coeff = np.repeat(np.identity(k_corr_clusters), repeats=n_series / k_corr_clusters, axis=0)

    z = np.array([__sample_from_distribution(dists[int(i / (n_series / k_corr_clusters / d_dist_clusters))
                                                   % d_dist_clusters + 1], t_points) for i in range(n_series)])
    x += beta * y_coeff @ y + (1 - rho_main - beta) * z

    return x
