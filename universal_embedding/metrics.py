from typing import List, Tuple, Union

import numpy as np


LabelType = Union[int, np.ndarray]
DomainType = Union[int, str]
NeighborType = Tuple[LabelType, DomainType]


def _to_1d_array(label: LabelType) -> np.ndarray:
    """Converts a label to a 1D numpy array."""
    arr = np.asarray(label)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def is_relevant(
    neighbor: NeighborType,
    query_label: LabelType,
    query_domain: DomainType,
) -> bool:
    """Returns True if neighbor shares at least one label and the same domain."""
    neighbor_label, neighbor_domain = neighbor[0], neighbor[1]

    query_labels = _to_1d_array(query_label)
    neighbor_labels = _to_1d_array(neighbor_label)

    label_match = np.isin(query_labels, neighbor_labels).any()
    domain_match = query_domain == neighbor_domain

    return bool(label_match and domain_match)


def is_domain_relevant(
    neighbor: NeighborType,
    query_domain: DomainType,
) -> bool:
    """Returns True if neighbor belongs to the same domain as the query."""
    neighbor_domain = neighbor[1]
    return bool(query_domain == neighbor_domain)


def universal_classif_accuracy(
    pred_label: LabelType,
    pred_domain: DomainType,
    query_label: LabelType,
    query_domain: DomainType,
) -> float:
    """Returns 1.0 if predicted label/domain are relevant to the query, else 0.0."""
    return 1.0 if is_relevant((pred_label, pred_domain), query_label, query_domain) else 0.0


def _prepare_relevances(
    neighbors: List[NeighborType],
    query_label: LabelType,
    query_domain: DomainType,
    top_k: int,
    throw_first: bool,
) -> List[int]:
    """Builds the binary relevance list up to top_k, optionally dropping rank 1."""
    limited_neighbors = neighbors[:top_k]
    relevances = [
        1 if is_relevant(neighbor, query_label, query_domain) else 0
        for neighbor in limited_neighbors
    ]

    if throw_first and relevances:
        relevances = relevances[1:]

    return relevances


def universal_mmp_at_k(
  top_k: int,
  actual_top_k: int,
  num_index_label: int,
  num_true_index_label: int,
  neighbors: List[NeighborType],
  query_label: LabelType,
  query_domain: DomainType,
  throw_first: bool
) -> Tuple[float, List[int]]:
  """
  Compute modified mean precision at k (mMP@K) for a single query.

  Iterates over the top neighbors (up to min(top_k, num_index_label)),
  counts the number of relevant neighbors, and normalizes by the minimum of
  num_true_index_label and actual_top_k. If throw_first is True, the top neighbor
  (assumed to be a self-match) is discarded.

  Parameters:
      top_k: Number of neighbors to consider.
      actual_top_k: Adjusted k after discarding (if throw_first is True).
      num_index_label: Total number of available neighbors.
      num_true_index_label: Total number of true positives in the index.
      neighbors: List of neighbor tuples (label, domain).
      query_label: The query label.
      query_domain: The query domain.
      throw_first: If True, discard the top neighbor.

  Returns:
      A tuple (mMP@K, relevances) where mMP@K is the computed metric (non-negative)
      and relevances is a list of binary values indicating which neighbors were relevant.
  """
  num_correct = 0
  relevances: List[int] = []

  # Check each neighbor up to the available limit.
  for j in range(min(top_k, num_index_label)):
    if is_relevant(neighbors[j], query_label, query_domain):
      num_correct += 1
      relevances.append(1)
    else:
      relevances.append(0)

  if throw_first:
    # Remove the top neighbor (assumed self-match) and ensure count is not negative.
    num_correct = max(num_correct - 1, 0)
    relevances = relevances[1:]

  # Compute mMP@K without penalizing queries with fewer positives.
  if num_true_index_label == 0: #edge case: no true positives
    mp = 0.0
  else:
    mp = (num_correct * 1.0) / min(num_true_index_label, actual_top_k)

  assert mp >= 0, "mMP@K should be non-negative."
  return mp, relevances


def universal_map_at_k(
    top_k: int,
    actual_top_k: int,
    num_true_index_label: int,
    neighbors: List[NeighborType],
    query_label: LabelType,
    query_domain: DomainType,
    throw_first: bool,
) -> Tuple[float, List[int]]:
    """Computes mean average precision at k for a single query."""
    relevances = _prepare_relevances(
        neighbors,
        query_label,
        query_domain,
        top_k,
        throw_first,
    )

    if len(relevances) != actual_top_k:
        raise ValueError(
            f"Length of relevances ({len(relevances)}) does not match "
            f"actual_top_k ({actual_top_k})."
        )

    if num_true_index_label == 0 or actual_top_k == 0:
        return 0.0, relevances

    relevances_arr = np.asarray(relevances, dtype=np.float32)
    cumsum_relevances = np.cumsum(relevances_arr)
    precision_at_rank = cumsum_relevances / (np.arange(len(relevances_arr)) + 1)

    average_precision = (
        (precision_at_rank * relevances_arr).sum()
        / min(num_true_index_label, actual_top_k)
    )

    return float(average_precision), relevances


def universal_recall_at_k(
    top_k: int,
    neighbors: List[NeighborType],
    query_label: LabelType,
    query_domain: DomainType,
    throw_first: bool,
) -> float:
    """Computes hit-based recall@k for a single query."""
    relevances = _prepare_relevances(
        neighbors,
        query_label,
        query_domain,
        top_k,
        throw_first,
    )
    return 1.0 if any(relevances) else 0.0