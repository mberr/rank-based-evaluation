"""Script to generate the plot #test alignments vs. #train alignments."""
import argparse
import logging

import mlflow
import pandas
import seaborn
from matplotlib import pyplot as plt


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_uri', type=str, default='http://localhost:5000')
    args = parser.parse_args()
    mlflow.set_tracking_uri(uri=args.tracking_uri)

    experiment_name = "adjusted_ranking_experiments"
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    if experiment is None:
        raise ValueError(f"Could not find experiment {experiment_name} at {args.tracking_uri}")

    df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
    )

    # select relevant columns
    rename = {
        "params.num_train_alignments": "#train",
        "params.num_test_alignments": "#test",
        "metrics.mean_rank": "MR",
        "metrics.hits_at_1": "H@1",
        "metrics.adjusted_mean_rank_index": "AMRI",
    }
    df = df.rename(columns=rename)
    order = [
        "MR",
        "H@1",
        "AMRI",
    ]

    # convert to int
    for col in ["#train", "#test"]:
        df[col] = pandas.to_numeric(df[col])

    plt.rc('font', size=14)
    fig, axes = plt.subplots(nrows=3, figsize=(8, 10), sharex=True)
    for y, ax in zip(order, axes):
        seaborn.lineplot(
            data=df,
            x="#test",
            y=y,
            hue="#train",
            legend=None if y != "H@1" else 'full',  # 'full',
            palette="viridis",
            ci=100,
            ax=ax,
        )
        ax.grid()
        if y == "MR":
            ax.set_ylim(1, None)
            ax.set_yscale('log')
        else:
            ax.set_ylim(0, 1)
    plt.xlim(df["#test"].min(), df["#test"].max())
    plt.subplots_adjust(wspace=None, hspace=None, left=.09, right=.95, bottom=0.07, top=.99)
    plt.savefig('eval.pdf')


if __name__ == '__main__':
    main()
