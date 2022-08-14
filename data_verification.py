import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.stats import spearmanr
from warnings import simplefilter
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List


def plot_pairwise_correlation_dist(*,
                                   axis: plt.Axes,
                                   empirical_matrix: np.array,
                                   synthetic_matrix: np.array,
                                   bins: int = 50) -> plt.Axes:
    """

    :param axis: A plt.Axes that you want to plot on
    :param empirical_matrix: A numpy array of empirical correlation matrices
    :param synthetic_matrix: A numpy array of synthetic correlation matrices
    :param bins: Number of bins for the histogram
    :return: Axes of the plot
    """

    empirical_matrix = empirical_matrix.reshape(empirical_matrix.size)
    synthetic_matrix = synthetic_matrix.reshape(synthetic_matrix.size)
    axis.hist(empirical_matrix, color='blue', density=True, bins=bins)
    axis.hist(synthetic_matrix, color='red', density=True, bins=bins)
    axis.legend(['EMP pairwise corr', 'SYN pairwise corr'])
    return axis


def plot_eigenvalues_distribution(*,
                                  axis: plt.Axes,
                                  empirical_matrix: np.array,
                                  synthetic_matrix: np.array,
                                  number_of_dominant_eigs: int = 1,
                                  bins: int = 10) -> plt.Axes:
    """

    :param axis: A plt.Axes that you want to plot on
    :param empirical_matrix: A numpy array of empirical correlation matrices
    :param synthetic_matrix: A numpy array of synthetic correlation matrices
    :param number_of_dominant_eigs: How many of the biggest eigenvalues of each correlation matrix to consider
    :param bins: Number of bins for the histogram
    :return: Axes of the plot
    """

    eev = np.real(np.linalg.eigvals(empirical_matrix))
    eev = np.sort(eev, axis=1)
    sev = np.real(np.linalg.eigvals(synthetic_matrix))
    sev = np.sort(sev, axis=1)

    eev = eev[:, -number_of_dominant_eigs:].reshape(eev[:, -number_of_dominant_eigs:].size)
    sev = sev[:, -number_of_dominant_eigs:].reshape(sev[:, -number_of_dominant_eigs:].size)

    axis.hist(eev, color='blue', density=True, bins=bins)
    axis.hist(sev, color='red', density=True, bins=bins)
    axis.legend(["EMP {} largest eigenval".format(number_of_dominant_eigs),
                 "SYN {} largest eigenval".format(number_of_dominant_eigs)])

    return axis


def plot_first_eigenvetors_entries_distribution(*,
                                                axis: plt.Axes,
                                                empirical_matrix: np.array,
                                                synthetic_matrix: np.array,
                                                bins: int = 20) -> plt.Axes:
    """

    :param axis: A plt.Axes that you want to plot on
    :param empirical_matrix: A numpy array of empirical correlation matrices
    :param synthetic_matrix: A numpy array of synthetic correlation matrices
    :param bins: Number of bins for the histogram
    :return: Axes of the plot
    """

    _, eev = np.linalg.eigh(empirical_matrix)
    _, sev = np.linalg.eigh(synthetic_matrix)
    eev = eev[:, :, -1].reshape(eev[:, :, -1].size)
    sev = sev[:, :, -1].reshape(sev[:, :, -1].size)
    axis.hist(eev, color='blue', density=True, bins=bins)
    axis.hist(sev, color='red', density=True, bins=bins)
    axis.legend(['EMP first eigenvec', 'SYN first eigenvec'])
    return axis


def __correlation2dist(corr: np.array) -> np.array:
    """

    :param corr: The correlation matrix
    :return: The distance matrix
    """

    dist = ((1 - corr) / 2) ** 0.5
    return dist


def __get_quasi_diag(link: np.array) -> list:
    """

    :param link: An array showing the links that were formed in the clustering process
    :return: A list of sorted clustered items by distance
    """

    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()


def __get_optimal_hierarchical(correlation_matrix: np.array, method: str = 'ward') -> np.array:
    """
    Reference: Advances in Financial Machine Learning, De Prado, 2018

    :param correlation_matrix: A numpy array of correlations
    :param method: A string of methods for clustering that can take value in ['single', 'complete', 'average',
    'weighted', 'centroid', 'median', 'ward']
    :return: A correlation matrix that is permuted in a way to show the hierarchical structure of correlations
    """

    correlation_matrix = pd.DataFrame(correlation_matrix, columns=range(correlation_matrix.shape[0]))
    dist = __correlation2dist(correlation_matrix)
    simplefilter("ignore", sch.ClusterWarning)
    link = sch.linkage(dist, method)
    sortIx = __get_quasi_diag(link)
    sortIx = correlation_matrix.index[sortIx].tolist()
    df0 = correlation_matrix.loc[sortIx, sortIx]
    return df0


def plot_optimal_hierarchical_cluster(*,
                                      axis: plt.Axes,
                                      correlation_matrix: np.array,
                                      method: str = 'ward',
                                      title: str = None) -> plt.Axes:
    """

    :param axis: A plt.Axes that you want to plot on
    :param correlation_matrix: A numpy array of the correlation matrix that we want it hierarchical structure
    :param method: The method of clustering that can take value in ['single', 'complete', 'average', 'weighted',
     'centroid', 'median', 'ward']
    :param title: A string for the title of the figure
    :return: A correlation matrix that is permuted in a way that show the hierarchical structure of correlations
    """

    sorted_corr_mx = __get_optimal_hierarchical(correlation_matrix, method)

    im = axis.imshow(sorted_corr_mx, origin='lower', vmin=0, vmax=1)
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

    if title is None:
        title = 'Optimal hierarchical structure'
    axis.set_title(title)

    return axis


def plot_hierarchical_structure(*,
                                axis_list: List[plt.Axes],
                                empirical_matrix: np.array,
                                synthetic_matrix: np.array) -> List[plt.Axes]:
    """

    :param axis_list: A list of plt.Axes with shape (2, n)
    :param empirical_matrix: Empirical correlation matrices with shape (n, dim, dim)
    :param synthetic_matrix: Synthetic correlation matrices with shape (n, dim, dim)
    :return: The list of plt.Axes
    """

    n = empirical_matrix.shape[0]
    for i in range(n):
        axis = axis_list[0, i]
        axis_list[0, i] = plot_optimal_hierarchical_cluster(axis=axis,
                                                            correlation_matrix=empirical_matrix[i, :, :],
                                                            title='EMP hierarchy')

        axis = axis_list[1, i]
        axis_list[1, i] = plot_optimal_hierarchical_cluster(axis=axis,
                                                            correlation_matrix=synthetic_matrix[i, :, :],
                                                            title='SYN hierarchy')

    return axis_list


def __get_mst(distance_matrix: np.array) -> nx.Graph:
    """

    :param distance_matrix: An array with shape (dim, dim) of distances between nodes
    :return: The MST of the graph based on distances
    """

    g = nx.Graph()
    edges = [(i, j, distance_matrix[i, j]) for (i, j) in np.ndindex(distance_matrix.shape) if j > i]
    g.add_weighted_edges_from(edges)
    g = g.to_undirected()
    mst = nx.minimum_spanning_tree(g)

    return mst


def plot_mst_degree_count(*,
                          axis_list: List[plt.Axes],
                          empirical_matrix: np.array,
                          synthetic_matrix: np.array) -> List[plt.Axes]:
    """
    Reference:  `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param axis_list: A list of plt.Axes that we want to plot on with two entries
    :param empirical_matrix: Empirical correlation matrices with shape (n, dim, dim)
    :param synthetic_matrix: Synthetic correlation matrices with shape (n, dim, dim)
    :return: The list of plt.Axes
    """

    N = empirical_matrix.shape[0]

    empirical_degrees = np.array([[val for (node, val) in __get_mst(__correlation2dist(empirical_matrix[i])).degree]
                                  for i in range(N)]).reshape(-1, 1)

    synthetic_degrees = np.array([[val for (node, val) in __get_mst(__correlation2dist(synthetic_matrix[i])).degree]
                                  for i in range(N)]).reshape(-1, 1)

    unique, counts = np.unique(empirical_degrees, return_counts=True)
    axis_list[0].loglog(unique, counts, 'o')
    axis_list[0].legend(['EMP log-log deg dist'])

    unique, counts = np.unique(synthetic_degrees, return_counts=True)
    axis_list[1].loglog(unique, counts, 'o')
    axis_list[1].legend(['SYN log-log deg dist'])

    return axis_list


def plot_stylized_facts(*,
                        empirical_matrix: np.array,
                        synthetic_matrix: np.array) -> None:
    """
    Reference: `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_


    :param empirical_matrix: Empirical correlation matrices with shape (n, dim, dim)
    :param synthetic_matrix: Synthetic correlation matrices with shape (n, dim, dim)
    :return: None
    """

    N, emx, smx = empirical_matrix.shape[0], empirical_matrix, synthetic_matrix
    fig, ax = plt.subplots(2, 4)

    plot_pairwise_correlation_dist(axis=ax[0, 0], empirical_matrix=emx, synthetic_matrix=smx)
    plot_eigenvalues_distribution(axis=ax[1, 0], empirical_matrix=emx, synthetic_matrix=smx, number_of_dominant_eigs=1)
    plot_eigenvalues_distribution(axis=ax[0, 1], empirical_matrix=emx, synthetic_matrix=smx, number_of_dominant_eigs=3)
    plot_first_eigenvetors_entries_distribution(axis=ax[1, 1], empirical_matrix=emx, synthetic_matrix=smx)
    plot_hierarchical_structure(axis_list=ax[:, 3][:, None], empirical_matrix=emx[0][None, :, :],
                                synthetic_matrix=smx[0][None, :, :])
    plot_mst_degree_count(axis_list=[ax[0, 2], ax[1, 2]], empirical_matrix=emx, synthetic_matrix=smx)

    plt.show()


def time_series_dependencies(*,
                                  time_series: np.array,
                                  dependence_method: str = 'gpr',
                                  tetha: float,
                                  **kwargs) -> np.array:
    """

    :param time_series: Time series with shape (N, T)
    :param dependence_method: The method of finding distance that can take value in ['gpr', 'gnpr']
    :param tetha: A float that specifies importance of each metrix in distance
    :param kwargs: It can contain h which specifies the size of each bin for non-parametric distribution modeling
    :return:
    """

    distance_matrix = np.zeros(shape=(time_series.shape[1], time_series.shape[1]))
    if dependence_method == 'gpr':
        means, stds = np.mean(time_series, axis=1), np.std(time_series, axis=1)
        corr = np.array(spearmanr(time_series, axis=1))[0, :, :]

        stds_mul = stds[:, None] @ stds[:, None].T
        stds_square = np.square(stds)
        sum_of_squared_stds = stds_square[:, None] + stds_square
        sq = np.sqrt(np.divide(2 * stds_mul, sum_of_squared_stds))
        exp = np.exp(-0.25 * np.divide(np.square(means[:, None] - means), sum_of_squared_stds))

        distance_matrix = tetha * (1 - corr) / 2 + (1 - tetha) * (1 - np.multiply(sq, exp))

    elif dependence_method == 'gnpr':
        h = kwargs['h']
        T = time_series.shape[1]
        sum_of_squared_difference = 3 / (T * (T * T - 1)) * np.sum(np.square(time_series - time_series[:, None]), axis=2)

        bins = np.arange(np.amin(time_series) - h, np.amax(time_series) + h, h)
        counts = np.array([np.histogram(time_series[i, :], bins=bins, density=True)[0]
                           for i in range(time_series.shape[0])])

        counts = np.sqrt(np.array(counts))
        sum_of_squared_difference_counts = np.sum(np.square(counts - counts[:, None]), axis=2) / 2

        distance_matrix = tetha * sum_of_squared_difference + (1 - tetha) * sum_of_squared_difference_counts

    return np.sqrt(distance_matrix)
