import numpy as np
from statsmodels.sandbox.distributions.multivariate import multivariate_t_rvs
from typing import Tuple


def generate_hcbm_correlation_matrix(*,
                                     n: int = 1,
                                     dim: int,
                                     block: int,
                                     depth: int,
                                     lower_bound: float = 0.1,
                                     upper_bound: float = 0.9,
                                     permute: bool = False) -> np.array:
    """
    Reference:  `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.
    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.
    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_

    :param n: Number of correlation matrices to generate
    :param dim: Dimension of each correlation matrix
    :param block: Number of blocks of correlated time series in each depth
    :param depth: The maximum level in hierarchical correlation structure
    :param lower_bound: The minimum amount of correlation possible
    :param upper_bound: The maximum amount of correlation possible
    :param permute: A boolean to specify if the output should be permuted
    :return:
    """

    corr_mxs = []
    for _ in range(n):
        rhos = np.random.uniform(lower_bound, upper_bound, size=depth)
        rhos = np.sort(np.append(rhos, [lower_bound, upper_bound]))

        correlation_matrix, current_depth = np.zeros(shape=(dim, dim)), 0
        correlation_matrix = __hcbm_recursive(correlation_matrix, (0, dim), block, depth, current_depth, rhos)
        np.fill_diagonal(correlation_matrix, 1)

        if permute:
            a = np.arange(dim)
            np.random.shuffle(a)
            correlation_matrix = correlation_matrix[a, :]
            correlation_matrix = correlation_matrix[:, a]

        corr_mxs.append(correlation_matrix)

    return np.array(corr_mxs)


def __hcbm_recursive(matrix: np.array,
                     block_range: Tuple[int],
                     block: int,
                     depth: int,
                     current_depth: int,
                     rhos: np.array) -> np.array:
    """

    :param matrix: A 2d array of the correlation matrix that we want to set values recursively with hcbm
    :param block_range: The first and last index of the randomly selected range that should be correlated
    :param block: The number of correlated time series in each depth
    :param depth: The maximum level in hierarchical correlation structure
    :param current_depth: The depth that we are in
    :param rhos: A list of ranges to choose correlation in their range based on current_depth
    :return: The correlation matrix with values set by hcbm
    """

    corr = np.random.uniform(rhos[current_depth], rhos[current_depth + 1])
    matrix[block_range[0]: block_range[1], block_range[0]: block_range[1]] = corr
    if current_depth == depth or block_range[0] == block_range[1]:
        return matrix

    blocks = np.random.choice(list(range(block_range[0], block_range[1])), block - 1)
    blocks = np.sort(np.append(blocks, [block_range[0], block_range[1]]))

    for i in range(len(blocks) - 1):
        matrix = __hcbm_recursive(matrix, (blocks[i], blocks[i + 1]), block, depth, current_depth + 1, rhos)

    return matrix


def time_series_from_correlation_matrix(*,
                                        correlation_matrix: np.array,
                                        t_points: int,
                                        distribution_type: str,
                                        deg_free: int = 3) -> np.array:
    """
    Reference:  `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.
    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.
    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_

    :param correlation_matrix: A numpy array of correlation matrix
    :param t_points: The number of samples for each time series
    :param distribution_type: The type of distribution that can take value in ['normal', 'student']
    :param deg_free: The degree of freedom for student distribution
    :return:
    """

    returns = np.array([])
    if distribution_type == 'normal':
        returns = np.random.multivariate_normal(mean=np.zeros(len(correlation_matrix)),
                                                cov=correlation_matrix,
                                                size=t_points)

    if distribution_type == 'student':
        returns = multivariate_t_rvs(m=np.zeros((len(correlation_matrix))),
                                     S=(deg_free - 2) / deg_free * correlation_matrix,
                                     df=deg_free,
                                     n=t_points)

    return returns.T
