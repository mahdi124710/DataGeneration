import numpy as np
import warnings
import math

warnings.filterwarnings('error')


def __generate_from_dvine(*, dim: int) -> np.array:
    """
    Reference: Generating random correlation matrices based partial correlations" by Harry Joe.
    https://www.sciencedirect.com/science/article/pii/S0047259X05000886

    :param dim: The dimension of the correlation matrix
    :return: The random correlation matrix generated with d-vine with shape (1, dim, dim)
    """

    beta, partial_correlations = dim / 2, np.ones(shape=(dim, dim))

    # 1. Generate partial correlation matrix
    for k in range(1, dim):
        i = np.arange(dim - k)
        partial_correlations[i, i + k] = 2 * np.random.beta(a=beta, b=beta, size=dim-k) - 1
        partial_correlations[i + k , i], beta = partial_correlations[i, i + k], beta - .5

    # 2. Calculate correlations
    for k in range(2, dim):
        for i in range(0, dim - k):
            j = i + k
            R = partial_correlations[i: j + 1, i: j + 1]
            r1, r3, R = R[0, 1: -1], R[-1, 1: -1], np.linalg.inv(R[1: -1, 1: -1])

            val, coef = r1 @ R @ r3.T, (1 - r1 @ R @ r1.T) * (1 - r3 @ R @ r3.T)

            partial_correlations[i, j] = partial_correlations[i, j] * np.sqrt(coef) + val
            partial_correlations[j, i] = partial_correlations[i, j]

    return np.expand_dims(partial_correlations, axis=0)


def __generate_from_cvine(*, dim: int, eta: float = None) -> np.array:
    """
    Reference: "Generating random correlation matrices based on vines and extended onion method"
    by Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
    https://www.sciencedirect.com/science/article/pii/S0047259X09000876

    :param dim: The dimension of the correlation matrix
    :param eta: The density of the correlation matrix will be proportional to [det(r)]^(eta - 1). Default is 2
    :return: A random correlation matrix generated with c-vine with shape (1, dim, dim)
    """

    eta = 2 if eta is None else eta
    beta, partial_correlations = eta + (dim - 1) / 2, np.ones(shape=(dim, dim))

    # 1. Generate matrix of partial correlations
    for k in range(dim - 1):
        beta -= .5
        partial_correlations[k, k + 1:] = 2 * np.random.beta(a=beta, b=beta, size=dim - k - 1) - 1

    # 2. Use the recursive formula. Implemented in vectorized operations
    for i in range(dim - 1, 0, -1):
        for j in range(i + 1, dim):
            partial_slice = np.vstack((partial_correlations[0:i, i], partial_correlations[0:i, j])).T
            mult_cumprod = np.cumprod(np.sqrt(np.prod(1 - np.square(partial_slice), axis=1)))
            sum_array = np.append(np.prod(partial_slice, axis=1), partial_correlations[i, j])
            partial_correlations[i, j] = np.sum(np.multiply(mult_cumprod, sum_array[1:])) + sum_array[0]

    # 3. Symmetrization
    i, j = np.triu_indices(dim, k=1)
    partial_correlations[j, i] = partial_correlations[i, j]

    return np.expand_dims(partial_correlations, axis=0)


def sample_from_vine(*,
                     n: int,
                     dim: int,
                     method: str,
                     eta: float = None) -> np.array:
    """

    :param n: Number of random correlation matrices
    :param dim: Dimension of each correlation matrix
    :param method: The vine method to use for generating random correlation matrices which can take value in ['cvine',
    'dvine']
    :param eta: eta value for the cvine method
    :return: Random correlation matrices with shape (n, dim, dim)
    """

    answer, matrix = np.zeros(shape=(1, dim, dim)), np.array([])

    for _ in range(n):
        if method == 'dvine':
            matrix = __generate_from_dvine(dim=dim)
        elif method == 'cvine':
            matrix = __generate_from_cvine(dim=dim, eta=eta)

        answer = np.concatenate((answer, matrix), axis=0)

    return answer[1:, :, :]


def sample_from_extended_onion(*,
                               n: int,
                               dim: int,
                               eta: float = 2) -> np.array:
    """
    Reference: "Generating random correlation matrices based on vines and extended onion method"
    by Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
    https://www.sciencedirect.com/science/article/pii/S0047259X09000876

    :param n: The number of random correlation matrices to generate
    :param dim: Dimension of correlation matrices
    :param eta: The eta parameter for onion method
    :return: Random correlation matrices with shape (n, shape, shape)
    """

    result = []

    def __find_A(matrix):
        u, s, vh = np.linalg.svd(matrix)
        return u @ np.sqrt(np.diag(s))

    for _ in range(n):
        beta = eta + (dim - 2) / 2
        corr = np.ones(shape=(2, 2))
        corr[0, 1] = corr[1, 0] = 2 * np.random.beta(a=beta, b=beta) - 1

        for i in range(dim - 2):
            beta -= .5
            y = np.random.beta((i + 2) / 2, b=beta)
            u = np.random.normal(0, 1, size=i + 2)
            u = u / np.linalg.norm(u)

            A = __find_A(corr)
            z = (math.sqrt(y) * A @ u)[:, None]
            corr = np.concatenate((corr, z), axis=1)
            z = np.append(z, 1)[None, :]
            corr = np.concatenate((corr, z), axis=0)

        result.append(corr)

    return np.array(result)
