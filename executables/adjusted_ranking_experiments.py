# coding=utf-8
"""Evaluation of different training and test sizes."""
import argparse
import logging
import random

import mlflow
import numpy
import torch
import tqdm

from kgm.data import get_dataset_by_name
from kgm.eval.matching import evaluate_matching_model
from kgm.models import GCNAlign
from kgm.modules import MarginLoss, SampledMatchingLoss, get_similarity
from kgm.training.matching import AlignmentModelTrainer
from kgm.utils.mlflow_utils import log_metrics_to_mlflow, log_params_to_mlflow


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dbp15k_jape')
    parser.add_argument('--subset', type=str, default='zh_en')
    parser.add_argument('--num_epochs', type=int, default=2_000)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--tracking_uri', type=str, default='http://localhost:5000')
    args = parser.parse_args()

    # Mlflow settings
    logging.info(f'Logging to MLFlow @ {args.tracking_uri}')
    mlflow.set_tracking_uri(uri=args.tracking_uri)
    mlflow.set_experiment('adjusted_ranking_experiments')

    # Determine device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device={device}")

    # load dataset
    dataset = get_dataset_by_name(
        dataset_name=args.dataset,
        subset_name=args.subset,
        inverse_triples=True,  # GCNAlign default
        self_loops=True,  # GCNAlign default
    )

    for num_train in [
        0,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        3000,
        5000,
        7500,
    ]:

        ea_full = dataset.alignment.all
        i_all = ea_full.shape[1]
        i_train = num_train

        # store optimal evaluation batch size for different sizes
        for iteration in tqdm.trange(args.iterations, unit='run', unit_scale=True):
            # fix random seed
            torch.manual_seed(iteration)
            numpy.random.seed(iteration)
            random.seed(iteration)

            # train-test split
            assert ea_full.shape[0] == 2
            ea_full = ea_full[:, torch.randperm(i_all)]
            ea_train, ea_test = ea_full[:, :i_train], ea_full[:, i_train:]

            # instantiate model
            model = GCNAlign(
                dataset=dataset,
                embedding_dim=200,
                n_layers=2,
                use_conv_weights=False,
            ).to(device=device)

            # instantiate similarity
            similarity = get_similarity(
                similarity="l1",
                transformation="negative",
            )

            if i_train > 0:
                # instantiate loss
                loss = SampledMatchingLoss(
                    similarity=similarity,
                    base_loss=MarginLoss(margin=3.),
                    num_negatives=50,
                )

                # instantiate trainer
                trainer = AlignmentModelTrainer(
                    model=model,
                    similarity=similarity,
                    dataset=dataset,
                    loss=loss,
                    optimizer_cls="adam",
                    optimizer_kwargs=dict(
                        lr=1.0,
                    ),
                )

                # train
                trainer.train(num_epochs=args.num_epochs)

            # evaluate with different test set sizes
            total_num_test_alignments = ea_test.shape[1]
            test_sizes = list(range(1_000, total_num_test_alignments, 1_000))
            results = dict(evaluate_matching_model(
                model=model,
                alignments={
                    k: ea_test[:, :k]
                    for k in test_sizes
                },
                similarity=similarity,
            )[0])

            # store results
            for size, result in results.items():
                # start experiment
                with mlflow.start_run():
                    log_params_to_mlflow(config=dict(
                        dataset=args.dataset,
                        subset=args.subset,
                        num_epochs=args.num_epochs,
                        num_train_alignments=i_train,
                        num_test_alignments=ea_test[:, :size].shape[1],
                        seed=iteration,
                    ))
                    log_metrics_to_mlflow(metrics=result)


if __name__ == '__main__':
    main()
