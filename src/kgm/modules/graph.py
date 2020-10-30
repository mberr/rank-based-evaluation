# coding=utf-8
"""
Module for message passing modules.

The message passing is split into three phases:
1) Message Creation
   Calculate messages. Potentially takes the source and target node representations, as well as the relation-type of
   the considered edge into account, i.e. for a triple (e_i, r, e_j): m_{i->j} = f(x_i, x_j, r)
2) Message Passing
   The message are exchanged, i.e. m_{i->j} moves from i to j. This is done in parallel for all messages.
3) Message Aggregation
   All incoming messages are aggregated into a single vector, i.e. a_j = agg({m_{i->j} for all i})
4) Node Update
   The new node representations are calculated given the aggregated messages, as well as the old node representation,
   i.e. x_j := update(x_j, a_j)
"""
import logging
from typing import Optional

import torch
from torch import nn

from ..utils.torch_utils import _guess_num_nodes
from ..utils.types import NodeIDs, RelationIDs

logger = logging.getLogger(name=__name__)

__all__ = [
    'AliGAT',
    'AliGate',
    'BasesLinearRelationSpecificMessageCreator',
    'BlockLinearRelationSpecificMessageCreator',
    'GAT',
    'GCNBlock',
    'IdentityMessageCreator',
    'LinearMessageCreator',
    'MeanAggregator',
    'MessagePassingBlock',
    'MessagePassingBlock',
    'OnlyUpdate',
    'SumAggregator',
]


class MissingEdgeTypesException(BaseException):
    """Class requires edge information."""

    def __init__(self, cls):
        super().__init__(f'{cls.__name__} requires passing edge types.')


# pylint: disable=abstract-method
class MessageCreator(nn.Module):
    """Abstract class for different methods to create messages to send."""

    def reset_parameters(self) -> None:
        """Reset the module's parameters."""
        # TODO: Subclass from ExtendedModule

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: torch.FloatTensor,
        source: NodeIDs,
        target: NodeIDs,
        edge_type: Optional[RelationIDs] = None,
    ) -> torch.FloatTensor:
        """
        Create messages.

        :param x: shape: (num_nodes, node_embedding_dim)
            The node representations.

        :param source: (num_edges,)
            The source indices for each edge.

        :param target: shape: (num_edges,)
            The target indices for each edge.

        :param edge_type: shape: (num_edges,)
            The edge type for each edge.

        :return: shape: (num_edges, message_dim)
            The messages source -> target.
        """
        raise NotImplementedError


class IdentityMessageCreator(MessageCreator):
    """Send source embeddings unchanged."""

    def forward(
        self,
        x: torch.FloatTensor,
        source: NodeIDs,
        target: NodeIDs,
        edge_type: Optional[RelationIDs] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        return x.index_select(dim=0, index=source)


class LinearMessageCreator(MessageCreator):
    """Transform source embeddings by learned linear transformation."""

    def __init__(
        self,
        input_dim: int,
        message_dim: int,
        use_bias: bool = False,
    ):
        """
        Initialize the message creator.

        :param input_dim: >0
            The number of input features, i.e. the dimension of the embedding vector.
        :param message_dim: > 0
            The number of output features, i.e. the dimension of the message vector.
        :param use_bias:
            Whether to use a bias after the linear transformation.
        """
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=message_dim, bias=use_bias)

    def reset_parameters(self) -> None:  # noqa: D102
        # TODO: Subclass from ExtendedModule
        self.linear.reset_parameters()

    def forward(
        self,
        x: torch.FloatTensor,
        source: NodeIDs,
        target: NodeIDs,
        edge_type: Optional[RelationIDs] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        x = self.linear(x)
        return x.index_select(dim=0, index=source)


# pylint: disable=abstract-method
class MessageAggregator(nn.Module):
    """
    Aggregation method for incoming messages.

    Should be permutation-invariant, and able to process an arbitrary number of messages into a single vector.
    """

    def reset_parameters(self) -> None:
        # TODO: Subclass from ExtendedModule
        pass

    # pylint: disable=arguments-differ
    def forward(
        self,
        msg: torch.FloatTensor,
        source: NodeIDs,
        target: NodeIDs,
        edge_type: Optional[RelationIDs] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.FloatTensor:
        """
        Aggregate messages per node.

        :param msg: shape: (num_edges, message_dim)
            The messages source -> target.

        :param source: (num_edges,)
            The source indices for each edge.

        :param target: shape: (num_edges,)
            The target indices for each edge.

        :param edge_type: shape: (num_edges,)
            The edge type for each edge.

        :param num_nodes: >0
            The number of nodes. If None is provided tries to guess the number of nodes by max(source.max(), target.max()) + 1

        :return: shape: (num_nodes, update_dim)
            The node updates.
        """
        raise NotImplementedError


class SumAggregator(MessageAggregator):
    """Sum over incoming messages."""

    def forward(
        self,
        msg: torch.FloatTensor,
        source: NodeIDs,
        target: NodeIDs,
        edge_type: Optional[RelationIDs] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        num_nodes = _guess_num_nodes(num_nodes=num_nodes, source=source, target=target)
        dim = msg.shape[1]
        return torch.zeros(num_nodes, dim, dtype=msg.dtype, device=msg.device).index_add_(dim=0, index=target, source=msg)


class MeanAggregator(MessageAggregator):
    """Average over incoming messages."""

    def forward(
        self,
        msg: torch.FloatTensor,
        source: NodeIDs,
        target: NodeIDs,
        edge_type: Optional[RelationIDs] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        num_nodes = _guess_num_nodes(num_nodes=num_nodes, source=source, target=target)
        dim = msg.shape[1]
        sum_agg = torch.zeros(num_nodes, dim, dtype=msg.dtype, device=msg.device).index_add_(dim=0, index=target, source=msg)
        uniq, count = torch.unique(target, sorted=False, return_counts=True)
        norm = torch.zeros(num_nodes, dtype=torch.long, device=msg.device).scatter_(dim=0, index=uniq, src=count).clamp_min(min=1).float().reciprocal().unsqueeze(dim=-1)
        return sum_agg * norm


# pylint: disable=abstract-method
class NodeUpdater(nn.Module):
    """Compute new node representation based on old representation and aggregated messages."""

    def reset_parameters(self) -> None:
        # TODO: Merge with AbstractKGMatchingModel's reset_parameters
        pass

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: torch.FloatTensor,
        delta: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Update node representations.

        :param x: shape: (num_nodes, node_embedding_dim)
            The node representations.

        :param delta: (num_nodes, update_dim)
            The node updates.

        :return: shape: (num_nodes, new_node_embedding_dim)
            The new node representations.
        """
        raise NotImplementedError


class OnlyUpdate(NodeUpdater):
    """Discard old node representation and only use aggregated messages."""

    def forward(
        self,
        x: torch.FloatTensor,
        delta: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return delta


# pylint: disable=abstract-method
class MessagePassingBlock(nn.Module):
    """A message passing block comprising a message creation, message aggregation, and an update module."""

    def __init__(
        self,
        message_creator: MessageCreator,
        message_aggregator: MessageAggregator,
        node_updater: NodeUpdater,
    ):
        """
        Initialize the block.

        :param message_creator:
            The module to create messages potentially based on the source and target node representation, as well as the
            edge type.
        :param message_aggregator:
            The module to aggregate all incoming messages to a fixed size vector.
        :param node_updater:
            The module to calculate the new node representation based on the old representation and the aggregated
            incoming messages.
        """
        super().__init__()

        # Bind sub-modules
        self.message_creator = message_creator
        self.message_aggregator = message_aggregator
        self.node_updater = node_updater

    def reset_parameters(self) -> None:
        """Reset parameters. Delegates to submodules."""
        self.message_creator.reset_parameters()
        self.message_aggregator.reset_parameters()
        self.node_updater.reset_parameters()

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: torch.FloatTensor,
        source: NodeIDs,
        target: NodeIDs,
        edge_type: Optional[RelationIDs] = None,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Calculate new node representations by message passing.

        :param x: shape: (num_nodes, node_embedding_dim)
            The node representations.

        :param source: (num_edges,)
            The source indices for each edge.

        :param target: shape: (num_edges,)
            The target indices for each edge.

        :param edge_type: shape: (num_edges,)
            The edge type for each edge.

        :param edge_weights: shape (num_edges,)
            The edge weights.

        :return: shape: (num_nodes, new_node_embedding_dim)
            The new node representations.
        """
        # create messages
        messages = self.message_creator(x=x, source=source, target=target, edge_type=edge_type)

        # apply edge weights
        if edge_weights is not None:
            messages = messages * edge_weights.unsqueeze(dim=-1)

        # aggregate
        delta = self.message_aggregator(msg=messages, source=source, target=target, edge_type=edge_type, num_nodes=x.shape[0])
        del messages

        return self.node_updater(x=x, delta=delta)


class GCNBlock(MessagePassingBlock):
    """
    GCN model roughly following https://arxiv.org/abs/1609.02907.

    Notice that this module does only the message passing part, and does **not** apply a non-linearity.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool,
    ):
        """
        Initialize the block.

        :param input_dim: >0
            The number of input features, i.e. the dimension of the embedding vector.
        :param output_dim: > 0
            The number of output features.
        :param use_bias:
            Whether to use a bias after the linear transformation.
        """
        super().__init__(
            message_creator=LinearMessageCreator(
                input_dim=input_dim,
                message_dim=output_dim,
                use_bias=use_bias
            ),
            message_aggregator=SumAggregator(),
            node_updater=OnlyUpdate()
        )
