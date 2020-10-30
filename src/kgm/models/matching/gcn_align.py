# coding=utf-8
"""
Implementation of GCN-Align.

The paper introducing the model can be found at https://www.aclweb.org/anthology/D18-1032.pdf.

The authors' implementation can be found at https://github.com/1049451037/GCN-Align and they also refer to
https://github.com/1049451037/HIN-Align for an improved implementation.
"""
import logging
from typing import Any, Mapping, Optional

import torch
from torch import nn

from .base import GraphBasedKGMatchingModel, IndependentSideMixin
from ...data import KnowledgeGraphAlignmentDataset, MatchSideEnum, SIDES
from ...data.reduction import DropRelationInformationKnowledgeGraphToGraphReduction, KnowledgeGraphToGraphReduction, target_normalization
from ...modules.embeddings.base import Embedding, EmbeddingNormalizationMode, NodeEmbeddingInitMethod, get_embedding_pair
from ...modules.embeddings.norm import EmbeddingNormalizationMethod
from ...modules.graph import GCNBlock, IdentityMessageCreator, MessagePassingBlock, OnlyUpdate, SumAggregator

logger = logging.getLogger(name=__name__)


class GCNAlign(IndependentSideMixin, GraphBasedKGMatchingModel):
    """GCN-Align model implementation."""

    #: The node embeddings
    node_embeddings: Mapping[MatchSideEnum, Embedding]

    def __init__(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        reduction_cls: Optional[KnowledgeGraphToGraphReduction] = None,
        reduction_kwargs: Optional[Mapping[str, Any]] = None,
        embedding_dim: int = 200,
        activation_cls: nn.Module = nn.ReLU,
        n_layers: int = 2,
        use_conv_weights: bool = False,
        node_embedding_init_method: NodeEmbeddingInitMethod = NodeEmbeddingInitMethod.sqrt_total,  # 'total',  # 'individual'
        vertical_sharing: bool = True,
        node_embedding_dropout: Optional[float] = None,
        node_embedding_init_config: Optional[Mapping[str, Any]] = None,
    ):
        """
        Initialize the model.

        :param dataset:
            The dataset.
        :param reduction_cls:
            The reduction strategy to obtain a (weighted) adjacency matrix from a knowledge graph.
        :param embedding_dim:
            The dimension of the node embedding.
        :param activation_cls:
            The non-linear activation to use between the message passing steps.
        :param n_layers:
            The number of layers.
        :param use_conv_weights:
            Whether to use convolution weights.
        :param node_embedding_init_method:
            The method used to initialize the node embeddings.
        :param vertical_sharing:
            Whether to use "vertical weight sharing", i.e. apply the same convolution weights for all layers.
        :param node_embedding_dropout:
            An optional dropout to use on the node embeddings.
        """
        if reduction_cls is None:
            reduction_cls = DropRelationInformationKnowledgeGraphToGraphReduction
            reduction_kwargs = dict(
                normalization=target_normalization,
            )
        super().__init__(dataset=dataset, reduction_cls=reduction_cls, reduction_kwargs=reduction_kwargs)

        # node embeddings
        self.node_embeddings = get_embedding_pair(
            init=node_embedding_init_method,
            dataset=dataset,
            embedding_dim=embedding_dim,
            dropout=node_embedding_dropout,
            trainable=True,
            init_config=node_embedding_init_config,
            norm=EmbeddingNormalizationMethod.l2,
            normalization_mode=EmbeddingNormalizationMode.every_forward,
        )

        # GCN layers
        self.n_layers = n_layers
        self.use_conv_weights = use_conv_weights
        self.vertical_sharing = vertical_sharing
        blocks = []
        if use_conv_weights:
            if self.vertical_sharing:
                gcn_block = GCNBlock(input_dim=embedding_dim, output_dim=embedding_dim, use_bias=True)
                activation = activation_cls()
                for _ in range(n_layers):
                    blocks.append(gcn_block)
                    blocks.append(activation)
            else:
                for _ in range(n_layers):
                    gcn_block = GCNBlock(input_dim=embedding_dim, output_dim=embedding_dim, use_bias=True)
                    activation = activation_cls()
                    blocks.append(gcn_block)
                    blocks.append(activation)
        else:
            message_block = MessagePassingBlock(
                message_creator=IdentityMessageCreator(),
                message_aggregator=SumAggregator(),
                node_updater=OnlyUpdate(),
            )
            for _ in range(n_layers):
                blocks.append(message_block)
                activation = activation_cls()
                blocks.append(activation)
        side_to_modules = {
            side: nn.ModuleList(blocks)
            for side in SIDES
        }
        self.layers = nn.ModuleDict(modules=side_to_modules)

        # Initialize parameters
        self.reset_parameters()

    def _forward_side(
        self,
        side: MatchSideEnum,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        x = self.node_embeddings[side](indices=None)

        # Prepare message passing keyword arguments
        adjacency = self.reductions[side]()
        message_passing_kwargs = {
            'source': adjacency.source,
            'target': adjacency.target,
            'edge_weights': adjacency.values,
        }

        # forward pass through all layers
        if side in self.layers.keys():
            layers = self.layers[side] if side in self.layers.keys() else []
        else:
            logger.warning('No layers for side %s', side)
            layers = []

        for layer in layers:
            if isinstance(layer, MessagePassingBlock):
                x = layer(x, **message_passing_kwargs)
            else:
                x = layer(x)

        # Select indices if requested
        if indices is not None:
            x = x[indices]

        return x
