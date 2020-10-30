"""Reduction strategies from Knowledge Graph to (weighted) uni-relational graphs."""
import enum
import logging
from typing import Callable, Optional

import torch

from .knowledge_graph import KnowledgeGraph
from ..utils.torch_utils import ExtendedModule, SparseCOOMatrix

logger = logging.getLogger(name=__name__)


# pylint: disable=abstract-method
class KnowledgeGraphToGraphReduction(ExtendedModule):
    r"""
    Base class for methods reducing the full KG tensor to a single adjacency matrix.

    A knowledge graph (KG) comprises a set of triples :math:`\mathcal{T} = \{(h, r, t)\}`, where
    :math:`h, r \in \mathcal{E}` are entities, and :math:`r \in \mathcal{R}` are relations.
    The KG can also be represenated by a three-dimensional binary tensor
    :math:`\mathbf{T} \in \{0, 1\}^{E \times R \times E}`, where :math:`E := |\mathcal{E}|`, and :math:`R := |\mathcal{R}|`.

    Often GCN-based models are only defined for uni-relational graphs. Thus, the KG adjacency tensor :math:`\mathbf{T}`
    needs to be reduced to a (weighted) adjacency matrix :math:`\mathbf{A} \in \mathbb{R}^{E \times E}`.
    """

    # pylint: disable=arguments-differ
    def forward(self) -> SparseCOOMatrix:
        """Get the (weighted) uni-relational adjacency matrix."""
        return self.get_adjacency()


def _get_raw_edge_tensor(knowledge_graph: KnowledgeGraph) -> torch.LongTensor:
    """Get the raw edge_tensor, i.e. {{(h,t) | (h,r,t) in T}}."""
    return knowledge_graph.triples[:, [0, 2]].t()


class StaticKnowledgeGraphToGraphReduction(KnowledgeGraphToGraphReduction):
    """A base class for parameter-free reduction."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        normalization: Optional[Callable[[SparseCOOMatrix], SparseCOOMatrix]] = None,
    ):
        """
        Initialize the reduction strategy.

        :param knowledge_graph:
            The knowledge graph to reduce.
        :param normalization:
            An optional normalization of the resulting adjacency matrix.
        """
        super().__init__()
        adjacency = self.get_static_adjacency(knowledge_graph=knowledge_graph)
        if normalization is not None:
            adjacency = normalization(adjacency)
        self.adjacency = adjacency

    def get_static_adjacency(self, knowledge_graph: KnowledgeGraph) -> SparseCOOMatrix:
        """Compute the adjacency matrix in advance."""
        raise NotImplementedError

    def forward(self) -> SparseCOOMatrix:  # noqa: D102
        return self.adjacency


# pylint: disable=abstract-method
class DropRelationInformationKnowledgeGraphToGraphReduction(StaticKnowledgeGraphToGraphReduction):
    """Drop the relation information, i.e. there is an edge if there is at least one triple."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        normalization: Optional[Callable[[SparseCOOMatrix], SparseCOOMatrix]] = None,
        unique: bool = True,
        add_self_loops: bool = False,
        add_inverse: bool = False,
    ):
        """
        Initialize the reduction strategy.

        :param knowledge_graph:
            The knowledge graph to reduce.
        :param normalization:
            An optional normalization of the resulting adjacency matrix.
        :param unique:
            Whether to drop duplicate edges.
        :param add_self_loops:
            Whether to add self-loops.
        :param add_inverse:
            Whether to add inverse edges, i.e. make the adjacency symmetric.
        """
        self.unique = unique
        self.add_self_loops = add_self_loops
        self.add_inverse = add_inverse
        super().__init__(knowledge_graph=knowledge_graph, normalization=normalization)

    def get_static_adjacency(self, knowledge_graph: KnowledgeGraph) -> SparseCOOMatrix:  # noqa: D102
        edge_tensor = _get_raw_edge_tensor(knowledge_graph)

        if self.add_inverse:
            edge_tensor = torch.cat([edge_tensor, edge_tensor.flip(0)], dim=1)
        if self.add_self_loops:
            edge_tensor = torch.cat([edge_tensor, torch.arange(knowledge_graph.num_entities, device=edge_tensor.device).view(1, -1).repeat(2, 1)], dim=-1)

        # Drop duplicates
        if self.unique:
            num_edges = edge_tensor.shape[1]
            edge_tensor = torch.unique(edge_tensor, dim=1)
            num_edges_reduced = edge_tensor.shape[1]
            if num_edges_reduced < num_edges:
                logger.info('Dropped %d/%d edges.', num_edges - num_edges_reduced, num_edges)
        return SparseCOOMatrix.from_edge_tensor(
            edge_tensor=edge_tensor,
            edge_weights=None,
            size=knowledge_graph.num_entities,
        )


def _scale_edge_weights(
    adjacency: SparseCOOMatrix,
    edge_factor: torch.FloatTensor,
) -> SparseCOOMatrix:
    """
    Multiply the edge weights by an edge-specific factor.

    Handles special case where the original matrix is unweighted.

    :param adjacency:
        The adjacency.
    :param edge_factor: shape: (num_edges,)
        The edge-wise factor.

    :return:
        The scaled adjacency matrix.
    """
    if adjacency.values is not None:
        edge_factor = adjacency.values * edge_factor
    return adjacency.with_weights(weights=edge_factor)


def target_normalization(adjacency: SparseCOOMatrix) -> SparseCOOMatrix:
    r"""
    Normalize an adjacency matrix row-wise.

    .. math ::
        \hat{A}_{ij} = A_{ij} / \sum_{k} A_{ik}

    :param adjacency:
        The adjacency matrix.

    :return:
        The normalized adjacency matrix.
    """
    return adjacency.normalize(dim=1)


def source_normalization(adjacency: SparseCOOMatrix) -> SparseCOOMatrix:
    r"""
    Normalize an adjacency matrix column-wise.

    .. math ::
        \hat{A}_{ij} = A_{ij} / \sum_{k} A_{kj}

    :param adjacency:
        The adjacency matrix.

    :return:
        The normalized adjacency matrix.
    """
    return adjacency.normalize(dim=0)


def symmetric_normalization(adjacency: SparseCOOMatrix) -> SparseCOOMatrix:
    r"""
    Normalize an adjacency matrix symmetrically.

    .. math ::
        \hat{A}_{ij} = A_{ij} / \sqrt{\left(\sum_{k} A_{kj} \right) \cdot \left(\sum_{k} A_{kj} \right)}

    :param adjacency:
        The adjacency matrix.

    :return:
        The normalized adjacency matrix.
    """
    edge_factor = (adjacency.scatter(adjacency.sum(dim=1), dim=0) * adjacency.scatter(adjacency.sum(dim=0), dim=1)).sqrt().reciprocal()
    # edge_factor = adjacency.scatter(adjacency.sum(dim=1).sqrt().reciprocal(), dim=0) * adjacency.scatter(adjacency.sum(dim=0).sqrt().reciprocal(), dim=1)
    return _scale_edge_weights(
        adjacency=adjacency,
        edge_factor=edge_factor,
    )


class EdgeWeightsEnum(str, enum.Enum):
    """Which edge weights to use."""

    #: None
    none = 'none'

    #: Inverse in-degree -> sum of weights for incoming messages = 1
    inverse_in_degree = 'inverse_in_degree'

    #: Inverse out-degree -> sum of weights for outgoing messages = 1
    inverse_out_degree = 'inverse_out_degree'

    #: 1 / sqrt(in-degree * out-degree)
    symmetric = 'symmetric'


def normalize_adjacency(
    adjacency: SparseCOOMatrix,
    mode: EdgeWeightsEnum,
) -> SparseCOOMatrix:
    """
    Normalize adjacency according to normalization mode.

    :param adjacency:
        The adjacency matrix.
    :param mode:
        The mode.

    :return:
        The normalized adjacency.
    """
    if mode == EdgeWeightsEnum.inverse_in_degree:
        return target_normalization(adjacency=adjacency)
    elif mode == EdgeWeightsEnum.inverse_out_degree:
        return source_normalization(adjacency=adjacency)
    elif mode == EdgeWeightsEnum.symmetric:
        return symmetric_normalization(adjacency=adjacency)
    return adjacency
