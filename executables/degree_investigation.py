"""Script to generate the evaluation for degree inductive bias."""
import numpy
import torch
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

from kgm.data import SIDES, get_dataset_by_name
from kgm.models import GCNAlign, PureEmbeddingModel


def degree_vs_norm(
    dataset
):
    # calculate degree
    degrees = dict()
    for i, side in enumerate(SIDES):
        graph = dataset.graphs[side]
        degree = torch.ones(graph.num_entities, dtype=torch.long)  # self-loops
        for col in [0, 2]:
            idx, cnt = torch.unique(graph.triples[:, col], return_counts=True)
            degree[idx] += cnt
        degrees[side] = degree

    # just random vectors
    pure_model = PureEmbeddingModel(
        dataset=dataset,
        embedding_dim=200,
    )

    # untrained gcn model on random vectors
    gcn_model = GCNAlign(
        dataset=dataset,
        embedding_dim=200,
        n_layers=2,
        use_conv_weights=False,
    )

    for label, model in dict(
        gcn=gcn_model,
        pure=pure_model,
    ).items():
        norm = {
            side: vectors.norm(dim=-1).detach().numpy()
            for side, vectors in model().items()
        }
        x, y = [], []
        for side, deg in degrees.items():
            x.append(deg)
            y.append(norm[side])
        x = numpy.concatenate(x)
        y = numpy.concatenate(y)
        print(label, spearmanr(y, x))


def degree_correlation(dataset):
    # compute degree for all aligned nodes
    degree = torch.empty_like(dataset.alignment.all)
    for i, side in enumerate(SIDES):
        graph = dataset.graphs[side]
        deg = torch.ones(graph.num_entities, dtype=torch.long)  # self-loops
        for col in [0, 2]:
            idx, cnt = torch.unique(graph.triples[:, col], return_counts=True)
            deg[idx] += cnt
        degree[i] = deg[dataset.alignment.all[i]]
    # compute correlation
    rho_p, p_p = pearsonr(*degree.numpy())
    rho_s, p_s = spearmanr(*degree.numpy())

    # plot
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.scatter(*degree.numpy(), marker=".", color="black")
    plt.yscale("log")
    plt.xscale("log")
    plt.axis("equal")
    plt.xlabel("degree " + dataset.graphs[SIDES[0]].lang_code + " [log]")
    plt.ylabel("degree " + dataset.graphs[SIDES[1]].lang_code + " [log]")
    plt.title(rf"Pearson $\rho$={rho_p:2.2%} (p={p_p}); Spearman $\rho$={rho_s:2.2%} (p={p_s})")
    plt.tight_layout()
    plt.savefig("degree_correlation.pdf")
    return degree


def main():
    # get dataset
    dataset = get_dataset_by_name(
        dataset_name='dbp15k_jape',
        subset_name='zh_en',
    )

    # degree correlation of aligned nodes
    degree_correlation(dataset=dataset)

    # degree vs. embedding norm
    degree_vs_norm(dataset=dataset)


if __name__ == '__main__':
    main()
