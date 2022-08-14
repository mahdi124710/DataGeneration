import tensorflow as tf
import numpy as np
from scipy.cluster import hierarchy
import fastcluster
from statsmodels.stats.correlation_tools import corr_nearest
import warnings

warnings.filterwarnings("ignore")


def sample_from_corrgan(*,
                        model_path: str,
                        n: int,
                        dim: int) -> np.array:
    """
    Refererence: `Marti, G., 2020, May. CorrGAN: Sampling Realistic Financial Correlation Matrices Using
    Generative Adversarial Networks. In ICASSP 2020-2020 IEEE International Conference on
    Acoustics, Speech and Signal Processing (ICASSP) (pp. 8459-8463). IEEE.
    <https://arxiv.org/pdf/1910.09504.pdf>`_

    :param model_path: A string showing the location that the pre-trained model is saved
    :param n: Number of correlation matrices to generate
    :param dim: Dimension of the square correlation matrices which takes value in [75, 80, 100, 120, 140, 160, 180, 200]
    :return: A numpy array with the shape=(n, dim, dim) containing n correlation matrices generated with the GAN
    """

    generator = tf.keras.models.load_model(model_path.format(dim))
    noise_dim = 100
    noise = tf.random.normal([n, noise_dim])
    generated_image = generator(noise, training=False)

    up_diagonal_x, up_diagonal_y = np.triu_indices(dim, k=1)
    nearest_correlation_matrices = []
    for i in range(n):
        corr_mat = np.array(generated_image[i, :, :, 0])
        np.fill_diagonal(corr_mat, 1)
        corr_mat[up_diagonal_y, up_diagonal_x] = corr_mat[up_diagonal_x, up_diagonal_y]

        nearest_corr_mat = corr_nearest(corr_mat)
        np.fill_diagonal(nearest_corr_mat, 1)
        nearest_corr_mat[up_diagonal_y, up_diagonal_x] = nearest_corr_mat[up_diagonal_x, up_diagonal_y]

        dist = 1 - nearest_corr_mat
        dim = len(dist)
        tri_a, tri_b = np.triu_indices(dim, k=1)
        Z = fastcluster.linkage(dist[tri_a, tri_b], method='ward')
        permutation = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, dist[tri_a, tri_b]))
        ordered_corr = nearest_corr_mat[permutation, :][:, permutation]

        nearest_correlation_matrices.append(ordered_corr)

    return np.array(nearest_correlation_matrices)
