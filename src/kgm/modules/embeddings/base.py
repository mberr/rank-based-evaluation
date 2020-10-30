"""Basic node embedding modules."""
import enum
import math
import pathlib
from typing import Any, Mapping, Optional, Type, Union

import torch
from torch import nn

from .init.base import ConstantNodeEmbeddingInitializer, NodeEmbeddingInitializer, RandomNodeEmbeddingInitializer
from .norm import EmbeddingNormalizationMethod, EmbeddingNormalizer, NoneEmbeddingNormalizer, get_normalizer_by_name
from ...data import KnowledgeGraph, KnowledgeGraphAlignmentDataset, MatchSideEnum
from ...utils.common import reduce_kwargs_for_method
from ...utils.torch_utils import ExtendedModule
from ...utils.types import NodeIDs


class EmbeddingNormalizationMode(str, enum.Enum):
    """The embedding normalization mode."""

    #: Do not normalize
    none = "none"

    #: Only normalize once after initialization
    initial = "initial"

    #: Normalize in every forward pass
    every_forward = "every_forward"

    #: Normalize after every parameter update (non-differentiable).
    after_update = "after_update"


# pylint: disable=abstract-method
class Embedding(ExtendedModule):
    """An embedding with additional initialization and normalization logic."""

    #: The actual data
    _embedding: nn.Embedding

    # The initializer
    initializer: NodeEmbeddingInitializer

    #: The normalizer
    normalizer: EmbeddingNormalizer

    #: additionally associated KnowledgeGraph
    graph: Optional[KnowledgeGraph]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: Optional[int] = None,
        initializer: Optional[NodeEmbeddingInitializer] = None,
        trainable: bool = True,
        normalizer: Optional[EmbeddingNormalizer] = None,
        normalization_mode: EmbeddingNormalizationMode = EmbeddingNormalizationMode.none,
        dropout: Optional[float] = None,
        shared: bool = False,
    ):
        """
        Initialize the module.

        :param num_embeddings:
            The number of embeddings.
        :param embedding_dim:
            The embedding dimension. If not provided, the initializer must provide one.
        :param initializer:
            The node embedding initializer.
        :param trainable:
            Whether the embeddings are trainable.
        :param normalizer:
            The node embedding normalizer.
        :param normalization_mode:
            The node embedding normalization mode.
        :param dropout:
            A node embedding dropout.
        :param shared:
            Whether to use a shared embedding for all nodes.
        """
        super().__init__()

        # Store embedding initialization method for re-initialization
        if initializer is None:
            initializer = RandomNodeEmbeddingInitializer()
        self.initializer = initializer

        if embedding_dim is None:
            embedding_dim = initializer.embedding_dim
        if embedding_dim is None:
            raise ValueError('Either embedding_dim must be provided, or the initializer must provide a dimension.')
        self.embedding_dim = embedding_dim

        if (normalization_mode == EmbeddingNormalizationMode.none) != (normalizer is None or isinstance(normalizer, NoneEmbeddingNormalizer)):
            raise ValueError("normalization_mode == none if and only if normalize is None.")
        if normalization_mode == EmbeddingNormalizationMode.after_update:
            raise NotImplementedError(normalization_mode)

        # Bind normalizer
        self.normalizer = normalizer
        self.normalization_mode = normalization_mode

        # Node embedding dropout
        if dropout is not None:
            dropout = nn.Dropout(p=dropout)
        self.dropout = dropout

        # Whether to share embeddings
        self.shared = shared

        # Store num nodes
        self.num_embeddings = num_embeddings

        # Allocate embeddings
        if self.shared:
            num_embeddings = 1
        self._embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        # Set trainability
        self._embedding.weight.requires_grad_(trainable)

        # Initialize
        self.reset_parameters()

    @property
    def weight(self) -> nn.Parameter:
        """Return the embedding weights."""
        return self._embedding.weight

    # pylint: disable=arguments-differ
    def forward(
        self,
        indices: Optional[NodeIDs] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass for embeddings.

        Optionally applies dropout and embedding normalization.

        :param indices:
            The indices to lookup. May be None to get all embeddings.

        :return: shape: (batch_size, embedding_dim)
            The embeddings. If indices=None, batch_size=num_embeddings.
        """
        if indices is None:
            if self.shared:
                x = self._embedding.weight.repeat(self.num_embeddings, 1)
            else:
                x = self._embedding.weight
        else:
            if self.shared:
                indices = torch.zeros_like(indices)
            x = self._embedding(indices)

        # apply dropout if requested
        if self.dropout is not None:
            x = self.dropout(x)

        # Apply normalization if requested
        if self.normalization_mode == EmbeddingNormalizationMode.every_forward:
            x = self.normalizer.normalize(x=x)

        return x

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters."""
        self.initializer.init_one_(embedding=self._embedding.weight)
        if self.normalization_mode in {
            EmbeddingNormalizationMode.initial,
            EmbeddingNormalizationMode.after_update
        }:
            self._embedding.weight.data = self.normalizer.normalize(x=self._embedding.weight.data)


class NodeEmbeddingInitMethod(str, enum.Enum):
    """Enum for selecting how to initialize node embeddings."""

    #: Initialize all to ones
    ones = 'ones'

    #: standard normal distribution
    std_one = 'std_one'

    #: std = 1 / sqrt(sum_i n_nodes_i)
    sqrt_total = 'sqrt_total'

    #: std = 1 / sqrt(n_nodes_i)
    sqrt_individual = 'sqrt_individual'

    def __str__(self):  # noqa: D105
        return str(self.name)


def get_embedding_pair(
    init: Union[NodeEmbeddingInitMethod, Type[NodeEmbeddingInitializer], NodeEmbeddingInitializer],
    dataset: KnowledgeGraphAlignmentDataset,
    embedding_dim: Optional[int] = None,
    dropout: Optional[float] = None,
    trainable: bool = True,
    init_config: Optional[Mapping[str, Any]] = None,
    norm: EmbeddingNormalizationMethod = EmbeddingNormalizationMethod.none,
    normalization_mode: EmbeddingNormalizationMode = EmbeddingNormalizationMode.none,
    shared: bool = False,
) -> Mapping[MatchSideEnum, Embedding]:
    """
    Create node embeddings for each graph side.

    :param init:
        The initializer. Can be a enum, a class, or an already initialized initializer.
    :param dataset:
        The dataset.
    :param embedding_dim:
        The embedding dimension. If not provided, the initializer must provide one.
    :param dropout:
        A node embedding dropout value.
    :param trainable:
        Whether the embedding should be set trainable.
    :param init_config:
        A key-value dictionary used for initializing the node embedding initializer (only relevant if not already
        initialized).
    :param norm:
        The embedding normalization method. The method is applied in every forward pass.
    :param normalization_mode:
        The node embedding normalization mode. None if and only if norm is None.
    :param shared:
        Whether to use one shared embedding for all nodes.

    :return:
        A mapping side -> node embedding.
    """
    # Build normalizer
    normalizer = get_normalizer_by_name(name=norm)

    return nn.ModuleDict({
        side: Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            initializer=resolve_initializer(
                init=init,
                dataset=dataset,
                side=side,
                init_config=init_config,
            ),
            trainable=trainable,
            normalizer=normalizer,
            normalization_mode=normalization_mode,
            dropout=dropout,
            shared=shared,
        )
        for side, num_embeddings in dataset.num_nodes.items()
    })


def init_method_normalizer(name: str):
    """Normalize the name of an initialization method."""
    return name.lower().replace('_', '').replace('nodeembeddinginitializer', '')


def resolve_initializer(
    init: Union[NodeEmbeddingInitMethod, Type[NodeEmbeddingInitializer], NodeEmbeddingInitializer, Mapping[MatchSideEnum, NodeEmbeddingInitializer]],
    dataset: KnowledgeGraphAlignmentDataset,
    side: MatchSideEnum,
    init_config: Optional[Mapping[str, Any]] = None,
    cache_root: pathlib.Path = None,
) -> NodeEmbeddingInitializer:
    """
    Resolve a node embedding intializer from a config.

    :param init:
        The chosen init. Can be
            * enum value
            * class
            * instance
            * mapping from side to instance.
    :param dataset:
        The dataset.
    :param side:
        The side for which the initializer should be created.
    :param init_config:
        Additional configuration for the initializer.
    :param cache_root:
        The cache root directory used for storing datasets. Defaults to ~/.kgm

    :return:
        An initializer instance.
    """
    if isinstance(init, dict):
        init = init[side]
    if cache_root is None:
        cache_root = pathlib.Path("~", ".kgm")
    cache_root = cache_root.expanduser()

    # already instantiated
    if isinstance(init, NodeEmbeddingInitializer):
        return init
    if isinstance(init, type) and issubclass(init, NodeEmbeddingInitializer):
        return init(**(reduce_kwargs_for_method(method=init.__init__, kwargs=init_config)))

    if init == NodeEmbeddingInitMethod.sqrt_total:
        total = sum(dataset.num_nodes.values())
        return RandomNodeEmbeddingInitializer(std=1. / math.sqrt(total))
    elif init == NodeEmbeddingInitMethod.sqrt_individual:
        return RandomNodeEmbeddingInitializer(std=1. / math.sqrt(dataset.num_nodes[side]))
    elif init == NodeEmbeddingInitMethod.std_one:
        return RandomNodeEmbeddingInitializer(std=1.)
    elif init == NodeEmbeddingInitMethod.ones:
        return ConstantNodeEmbeddingInitializer(value=1.0)
    else:
        raise ValueError(init)
