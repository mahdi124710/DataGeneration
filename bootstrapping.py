import numpy as np
from typing import Tuple


def row_bootstrap(*,
                  matrix: np.array,
                  n_samples: int = 1,
                  size: Tuple[int, int] = None) -> np.array:
    """
    Reference:  `Musciotto, F., Marotta, L., Miccichè, S. and Mantegna, R.N., 2018. Bootstrap validation of
    links of a minimum spanning tree. Physica A: Statistical Mechanics and its Applications,
    512, pp.1032-1043. <https://arxiv.org/pdf/1802.03395.pdf>`_.

    :param matrix: A 2d matrix that we want to bootstrap
    :param n_samples: The number of output bootstrapped matrices
    :param size: A tuple to show the dimensions of each bootstrapped matrix
    :return: A numpy array with shape (n_samples, size[0], size[1])
    """

    if size is None:
        size = matrix.shape

    bootstrapped_matrices = []
    for _ in range(n_samples):
        sr = selected_rows = np.random.choice(matrix.shape[0], size[0], replace=True)
        sc = selected_columns = np.random.choice(matrix.shape[1] - size[1] + 1, size[0], replace=True)
        mx = np.array([matrix[sr[i], sc[i]: sc[i] + size[1]] for i in range(len(sc))])
        bootstrapped_matrices.append(mx)

    return np.array(bootstrapped_matrices)


def pair_bootstrap(*,
                   matrix: np.array,
                   n_samples: int = 1,
                   size: int) -> np.array:
    """
    Reference: `Musciotto, F., Marotta, L., Miccichè, S. and Mantegna, R.N., 2018. Bootstrap validation of
    links of a minimum spanning tree. Physica A: Statistical Mechanics and its Applications,
    512, pp.1032-1043. <https://arxiv.org/pdf/1802.03395.pdf>`_.

    :param matrix: A time series matrix with shape T by N
    :param n_samples: The number of bootstrapped correlation matrices
    :param size: The number of random sampling for each pair
    :return: A numpy array with shape=(n_samples, N, N) of bootstrapped correlation matrices
    """

    N, T = matrix.shape[1], matrix.shape[0]
    corr_mxs = np.zeros(shape=(n_samples, N, N))
    pair1, pair2 = np.triu_indices(N, k=1)

    for i in range(len(pair1)):
        pair = row_bootstrap(matrix=matrix[:, [pair1[i], pair2[i]]], size=[size, 2], n_samples=n_samples)
        corr = np.array([np.corrcoef(pair[i, :, :], rowvar=False) for i in range(n_samples)])
        corr_mxs[:, pair1[i], pair2[i]] = corr[:, 0, 1]

    corr_mxs[:, pair2, pair1] = corr_mxs[:, pair1, pair2]
    [np.fill_diagonal(corr_mxs[i, :, :], 1) for i in range(n_samples)]

    return np.array(corr_mxs)


def block_bootstrap(*,
                    matrix: np.array,
                    n_samples: int,
                    size: Tuple[int],
                    block_size: Tuple[int]) -> np.array:
    """
    `Künsch, H.R., 1989. The jackknife and the bootstrap for general stationary observations.
    Annals of Statistics, 17(3), pp.1217-1241. <https://projecteuclid.org/euclid.aos/1176347265>`_.

    :param matrix: A 2d data matrix that we want to be bootstrapped
    :param n_samples: The number of bootstrapped matrices
    :param size: The dimensions of each bootstrapped matrix
    :param block_size: The dimensions of each randomly selected block
    :return: An array with shape=(n_samples, size[0], size[1]) of bootstrapped matrices
    """

    candidate_rows = np.arange(0, matrix.shape[0], block_size[0])[:-1]
    candidate_rows = np.append(candidate_rows, matrix.shape[0] - block_size[0])

    candidate_columns = np.arange(0, matrix.shape[1], block_size[1])[:-1]
    candidate_columns = np.append(candidate_columns, matrix.shape[1] - block_size[1])

    bnc = block_number_in_col = int(size[0] / block_size[0])
    bnr = block_number_in_row = int(size[1] / block_size[1])
    nb = number_of_blocks_in_each_bootstrapped_matrix = bnr * bnc
    b0, b1 = block_size[0], block_size[1]

    rows = np.random.choice(candidate_rows, bnr * bnc * n_samples, replace=True)
    cols = np.random.choice(candidate_columns, bnr * bnc * n_samples, replace=True)

    bm = np.block(
        [[[matrix[rows[k * nb + j * bnr + i]: rows[k * nb + j * bnr + i] + b0,
           cols[k * nb + j * bnr + i]: cols[k * nb + j * bnr + i] + b1]
           for i in range(bnr)] for j in range(bnc)] for k in range(n_samples)]
    )

    return bm
