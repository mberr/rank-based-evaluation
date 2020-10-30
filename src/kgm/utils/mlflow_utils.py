# coding=utf-8
"""Utility methods for MLFlow."""
import hashlib
import itertools
import logging
import os
import platform
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Tuple, Union

import mlflow
import mlflow.entities
import pandas
import tqdm

from .common import to_dot

logger = logging.getLogger(name=__name__)


def log_params_to_mlflow(
    config: Dict[str, Any],
) -> None:
    """Log parameters to MLFlow. Allows nested dictionaries."""
    nice_config = to_dot(config)
    # mlflow can only process 100 parameters at once
    keys = sorted(nice_config.keys())
    batch_size = 100
    for start in range(0, len(keys), batch_size):
        mlflow.log_params({k: nice_config[k] for k in keys[start:start + batch_size]})


def log_metrics_to_mlflow(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    """Log metrics to MLFlow. Allows nested dictionaries."""
    nice_metrics = to_dot(metrics, prefix=prefix)
    mlflow.log_metrics(nice_metrics, step=step)


def query_mlflow(
    tracking_uri: str,
    experiment_id: str,
    params: Dict[str, Union[str, int, float]] = None,
    metrics: Dict[str, Union[str, int, float]] = None,
    tags: Dict[str, Union[str, int, float]] = None
) -> List[mlflow.entities.Run]:
    """Query MLFlow for runs with matching params, metrics and tags."""
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # Construct query
    q_params = [f'params.{p} = "{v}"' for p, v in to_dot(params).items()] if params else []
    q_metrics = [f'metrics.{m} = "{v}"' for m, v in to_dot(metrics).items()] if metrics else []
    q_tags = [f'tags.{t} = "{v}"' for t, v in tags.items()] if tags else []
    query = ' and '.join([*q_params, *q_metrics, *q_tags])

    return client.search_runs(experiment_id, query)


def experiment_name_to_id(
    tracking_uri: str,
    experiment_id: int,
) -> str:
    """Convert an experiment name to experiment ID."""
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    return [exp.name for exp in client.list_experiments() if int(exp.experiment_id) == experiment_id][0]


def get_metric_history_for_runs(
    tracking_uri: str,
    metrics: Union[str, Collection[str]],
    runs: Union[str, Collection[str]],
) -> pandas.DataFrame:
    """
    Get metric history for selected runs.

    :param tracking_uri:
        The URI of the tracking server.
    :param metrics:
        The metrics.
    :param runs:
        The IDs of selected runs.

    :return:
         A dataframe with columns {'run_id', 'key', 'step', 'timestamp', 'value'}.
    """
    # normalize input
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(runs, str):
        runs = [runs]
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    data = []
    task_list = sorted(itertools.product(metrics, runs))
    n_success = n_error = 0
    with tqdm.tqdm(task_list, unit='metric+task', unit_scale=True) as progress:
        for metric, run in progress:
            try:
                data.extend(
                    (run, measurement.key, measurement.step, measurement.timestamp, measurement.value)
                    for measurement in client.get_metric_history(run_id=run, key=metric)
                )
                n_success += 1
            except ConnectionError as error:
                n_error += 1
                progress.write(f'[Error] {error.strerror}')
            progress.set_postfix(dict(success=n_success, error=n_error))
    return pandas.DataFrame(
        data=data,
        columns=['run_id', 'key', 'step', 'timestamp', 'value']
    )


def get_metric_history(
    tracking_uri: str,
    experiment_ids: Union[int, Collection[int]],
    metrics: Collection[str],
    runs: Optional[Collection[str]] = None,
    convert_to_wide_format: bool = False,
    filter_string: Optional[str] = "",
) -> pandas.DataFrame:
    """
    Get metric history data for experiment(s).

    :param tracking_uri:
        The URI of the tracking server.
    :param experiment_ids:
        The experiments ID(s).
    :param metrics:
        The name of the metrics to retrieve the history for.
    :param runs:
        An optional selection of runs via IDs. If None, get all.
    :param convert_to_wide_format:
        Whether to convert the dataframe from "long" to "wide" format.
    :param filter_string:
        Filter query string, defaults to searching all runs.

    :return:
        A dataframe of results.
    """
    # Normalize runs
    if runs is None:
        runs = get_all_runs_from_experiments(
            tracking_uri=tracking_uri,
            filter_string=filter_string,
            experiment_ids=experiment_ids
        )
        logger.info(f'Retrieved {len(runs)} runs for experiment(s) {experiment_ids}.')
    df = get_metric_history_for_runs(tracking_uri=tracking_uri, metrics=metrics, runs=runs)
    if convert_to_wide_format:
        df = _convert_metric_history_long_to_wide(history_df=df)
    return df


def _convert_metric_history_long_to_wide(
    history_df: pandas.DataFrame,
) -> pandas.DataFrame:
    """
    Convert ta dataframe of metric history from "long" to "wide" format.

    :param history_df:
        The dataframe in long format.

    :return:
        The dataframe in wide format.
    """
    return history_df.pivot_table(
        index=['run_id', 'step'],
        values='value',
        columns=['key'],
    )


def get_all_runs_from_experiments(
    *,
    experiment_ids: Union[int, Collection[int]],
    filter_string: Optional[str] = "",
    tracking_uri: Optional[str] = None,
    client: Optional[mlflow.tracking.MlflowClient] = None,
) -> Collection[str]:
    """
    Collect IDs for all runs associated with an experiment ID.

    .. note ::
        Exactly one of `tracking_uri` or `client` has to be provided.

    :param experiment_ids:
        The experiment IDs.
    :param filter_string:
        Filter query string, defaults to searching all runs.
    :param tracking_uri:
        The Mlflow tracking URI.
    :param client:
        The Mlflow client.


    :return:
        A collection of run IDs.
    """
    # Normalize input
    if isinstance(experiment_ids, int):
        experiment_ids = [experiment_ids]
    if None not in {tracking_uri, client}:
        raise ValueError('Cannot provide tracking_uri and client.')
    if tracking_uri is not None:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    runs = []

    # support for paginated results
    continue_searching = True
    page_token = None

    while continue_searching:
        page_result_list = client.search_runs(
            experiment_ids=list(map(str, experiment_ids)),
            filter_string=filter_string,
            page_token=page_token
        )
        runs.extend(run.info.run_uuid for run in page_result_list)
        page_token = page_result_list.token
        continue_searching = page_token is not None

    return runs


def _sort_key(x: Mapping[str, Any]) -> str:
    return hashlib.md5((';'.join(f'{k}={x}' for k, v in x.items()) + ';' + str(platform.node()) + ';' + str(os.getenv('CUDA_VISIBLE_DEVICES', '?'))).encode()).hexdigest()


def run_experiments(
    search_list: List[Mapping[str, Any]],
    experiment: Callable[[Mapping[str, Any]], Tuple[Mapping[str, Any], int]],
    num_replicates: int = 1,
    break_on_error: bool = False,
) -> None:
    """
    Run experiments synchronized by MLFlow.

    :param search_list:
        The search list of parameters. Each entry corresponds to one experiment.
    :param experiment:
        The experiment as callable. Takes the dictionary of parameters as input, and produces a result dictionary as well as a final step.
    """
    # randomize sort order to avoid collisions with multiple workers
    search_list = sorted(search_list, key=_sort_key)

    n_experiments = len(search_list)
    counter = {
        'error': 0,
        'success': 0,
        'skip': 0,
    }
    for run, params in enumerate(search_list * num_replicates):
        logger.info('================== Run %4d/%4d ==================', run, n_experiments * num_replicates)
        params = dict(**params)

        # Check, if run with current parameters already exists
        query = ' and '.join(list(map(lambda item: f"params.{item[0]} = '{str(item[1])}'", to_dot(params).items())))
        logger.info('Query: \n%s\n', query)

        run_hash = hashlib.md5(query.encode()).hexdigest()
        params['run_hash'] = run_hash
        logger.info('Hash: %s', run_hash)

        existing_runs = mlflow.search_runs(filter_string=f"params.run_hash = '{run_hash}'", run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY)
        if len(existing_runs) >= num_replicates:
            logger.info('Skipping existing run.')
            counter['skip'] += 1
            continue

        mlflow.start_run()

        params['environment'] = {
            'server': platform.node(),
        }

        # Log to MLFlow
        log_params_to_mlflow(params)
        log_metrics_to_mlflow({'finished': False}, step=0)

        # Run experiment
        try:
            final_evaluation, final_step = experiment(params)
            # Log to MLFlow
            log_metrics_to_mlflow(metrics=final_evaluation, step=final_step)
            log_metrics_to_mlflow({'finished': True}, step=final_step)
            counter['success'] += 1
        except Exception as e:  # pylint: disable=broad-except
            logger.error('Error occured.')
            logger.exception(e)
            log_metrics_to_mlflow(metrics={'error': 1})
            counter['error'] += 1
            if break_on_error:
                raise e

        mlflow.end_run()

    logger.info('Ran %d experiments.', counter)
