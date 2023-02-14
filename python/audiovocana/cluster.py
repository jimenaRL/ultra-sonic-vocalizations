from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.cluster as sc
import scipy as sp

import seaborn as sns
import matplotlib.pyplot as plt

from audiovocana.color import (
    COLORS,
    dec
)


def get_colors(metadata, keys_colors):
    colors = [metadata[k].map(COLORS[k]) for k in keys_colors]
    colors = pd.concat(colors, axis=1)
    colors.columns = keys_colors
    return colors


def get_feats_and_metadata(dataset, feats_names, metadata_keys):
    """
    return feats, metadata
    """

    metadata_records = []
    feats = {key: [] for key in feats_names}

    for sample in tqdm(dataset):

        metadata_records.append({
            k: dec(sample[k].numpy()) for k in metadata_keys
        })

        for feat_name in feats_names:
            feats[feat_name].append(sample[feat_name].numpy())

    for feat_name in feats_names:
        if feat_name[:5] == 'mean_' or feat_name[:4] == 'max_':
            feats[feat_name] = np.array(feats[feat_name]).T

    metadata = pd.DataFrame.from_records(metadata_records)

    return feats, metadata


def hcluster(
    X,
    colors,
    min_clusters=2,
    max_clusters=8,
    method='ward',
    metric="euclidean",
    feat='mean_stft',
    nb_cl_labels=4,
    scale='standard',
    figsize=(12, 12)):

    if scale == 'standard':
        standard_scale = 0
        z_score = None
        print('Using max-0 scaling')
    elif scale == 'whitening':
        standard_scale = None
        z_score = 0
        print('Using whitening scaling')
    else:
        raise ValueError

    cluster_assigns = {}

    g = sns.clustermap(
        pd.DataFrame(X),
        method=method,
        metric=metric,
        # Max-0 scaling. Either 0 (rows) or 1 (columns) or None
        standard_scale=standard_scale,
        # Whitening. Either 0 (by rows) or 1 (by columns) or None
        z_score=z_score,
        row_cluster=False,
        col_colors=colors,
        # Value at which to center the colormap when plotting divergent data.
        center=None,
        figsize=(16, 8),
        cbar_pos=None,
        xticklabels=1,
        yticklabels='auto'
    )

    plt.close('all')

    # linkage matrix
    Z = g.dendrogram_col.linkage
    for n in range(min_clusters, max_clusters + 1):
        n_assigs = sc.hierarchy.fcluster(Z, t=n, criterion='maxclust')
        cluster_assigns[n] = n_assigs
        clusters = range(1, n + 1)
        dicc_clusters = dict(zip(
            clusters,
            sns.color_palette("cubehelix", n)))
        colors_clusters = pd.DataFrame(n_assigs)[0].map(dicc_clusters)

        tmp_colors = [f'{n}-clusters-labels'] + colors.columns.tolist()
        colors = pd.concat([colors_clusters, colors], axis=1)
        colors.columns = tmp_colors

    g = sns.clustermap(
        pd.DataFrame(X),
        method=method,
        metric=metric,
        # Max-0 scaling. Either 0 (rows) or 1 (columns) or None
        standard_scale=standard_scale,
        # Whitening. Either 0 (by rows) or 1 (by columns) or None
        z_score=z_score,
        row_cluster=False,
        col_colors=colors,
        # Value at which to center the colormap when plotting divergent data.
        center=None,
        figsize=figsize,
        cbar_pos=None,
        xticklabels=7,
        yticklabels='auto')

    re_ind = g.dendrogram_col.reordered_ind
    g.ax_heatmap.axes.set_xticklabels(
        cluster_assigns[nb_cl_labels][re_ind][np.arange(0, len(re_ind), 7)]
    )
    for tick in g.ax_heatmap.axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
        tick.label.set_rotation('horizontal')

    # g.ax_heatmap.axes.set_yticklabels(
    #     get_ylabels()
    # )

    for tick in g.ax_heatmap.axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
        tick.label.set_rotation('horizontal')

    return g, cluster_assigns
