"""Common training loop parts."""
import logging
from typing import Any, Generic, Iterable, Mapping, Optional, Tuple, Type, TypeVar

import torch
from torch import nn
from torch.optim import Optimizer

from kgm.utils.common import NonFiniteLossError, kwargs_or_empty, last
from kgm.utils.torch_utils import construct_optimizer_from_config, get_device

logger = logging.getLogger(name=__name__)

BatchType = TypeVar('BatchType')


class BaseTrainer(Generic[BatchType]):
    """A base class for training loops."""

    #: The model
    model: nn.Module

    #: The optimizer instance
    optimizer: Optimizer

    def __init__(
        self,
        model: nn.Module,
        train_batch_size: Optional[int] = None,
        optimizer_cls: Type[Optimizer] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        clip_grad_norm: Optional[float] = None,
        accumulate_gradients: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize a new training loop.

        :param model:
            The model to train.
        :param train_batch_size:
            The batch size to use for training.
        :param optimizer_cls:
            The optimizer class.
        :param optimizer_kwargs:
            Keyword-based arguments for the optimizer.
        :param clip_grad_norm:
            Whether to apply gradient clipping (norm-based).
        :param accumulate_gradients:
            Accumulate gradients over batches. This can be used to simulate a larger batch size, while keeping the
            memory footprint small.
        :param device:
            The device on which to train.
        :param accumulate_gradients:
            Accumulate gradients over batches. This can be used to simulate a larger batch size, while keeping the
            memory footprint small.
        :param device:
            The device on which to train.
        """
        device = get_device(device=device)
        # Bind parameters
        self.train_batch_size = train_batch_size
        self.model = model.to(device=device)
        self.epoch = 0
        self.accumulate_gradients = accumulate_gradients
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.accumulate_gradients = accumulate_gradients
        self.device = device

        # create optimizer
        if optimizer_cls is None:
            optimizer_cls = 'adam'
        optimizer_config = dict(cls=optimizer_cls)
        optimizer_config.update(kwargs_or_empty(optimizer_kwargs))
        self.optimizer_config = optimizer_config
        self.reset_optimizer()

    def reset_optimizer(self) -> None:
        """Reset the optimizer."""
        self.optimizer = construct_optimizer_from_config(
            model=self.model,
            optimizer_config=self.optimizer_config,
        )

    def _train_one_epoch(self) -> Mapping[str, Any]:
        """
        Train the model for one epoch on the given device.

        :return:
            A dictionary of training results. Contains at least `loss` with the epoch loss value.
        """
        epoch_loss, counter = 0., 0

        # Iterate over batches
        i = -1
        for i, batch in enumerate(self._iter_batches()):
            # Compute batch loss
            batch_loss, real_batch_size = self._train_one_batch(batch=batch)

            # Break on non-finite loss values
            if not torch.isfinite(batch_loss).item():
                raise NonFiniteLossError

            # Update epoch loss
            epoch_loss += batch_loss.item() * real_batch_size
            counter += real_batch_size

            # compute gradients
            batch_loss.backward()

            # Apply gradient updates
            if i % self.accumulate_gradients == 0:
                self._parameter_update()

        # For the last batch, we definitely do an update
        if self.accumulate_gradients > 1 and (i % self.accumulate_gradients) != 0:
            self._parameter_update()

        return dict(
            loss=epoch_loss / counter
        )

    def _parameter_update(self):
        """Update the parameters using the optimizer."""
        # Gradient clipping
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                parameters=(p for p in self.model.parameters() if p.requires_grad),
                max_norm=self.clip_grad_norm,
            )

        # update parameters
        self.optimizer.step()

        # clear gradients afterwards
        self.optimizer.zero_grad()

    def _iter_batches(self) -> Iterable[BatchType]:
        """Iterate over batches."""
        raise NotImplementedError

    def _train_one_batch(self, batch: BatchType) -> Tuple[torch.Tensor, int]:
        """
        Train on a single batch.

        :param batch: shape: (batch_size,)
            The sample IDs.

        :return:
            A tuple (batch_loss, real_batch_size) of the batch loss (a scalar tensor), and the actual batch size.
        """
        raise NotImplementedError

    def train_iter(
        self,
        num_epochs: int = 1,
    ) -> Iterable[Mapping[str, Any]]:
        """
        Train the model, and return intermediate results.

        :param num_epochs:
            The number of epochs.
        :return:
            One result dictionary per epoch.
        """
        epoch_result = dict()
        for _ in range(self.epoch, self.epoch + num_epochs):
            self.model.train()

            # training step
            self.epoch += 1
            epoch_result = dict(
                epoch=self.epoch,
                train=self._train_one_epoch(),
            )

            yield epoch_result

        return epoch_result

    def train(
        self,
        num_epochs: int = 1,
        final_eval: bool = True,
    ) -> Mapping[str, Any]:
        """
        Train the model, and return intermediate results.

        :param num_epochs:
            The number of epochs.
        :param final_eval:
            Whether to perform an evaluation after the last training epoch.

        :return:
            A dictionary containing the result.
        """
        return last(self.train_iter(num_epochs=num_epochs))
