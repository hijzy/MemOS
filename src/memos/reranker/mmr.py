# memos/reranker/mmr.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memos.log import get_logger
from memos.utils import timed

from .base import BaseReranker

if TYPE_CHECKING:
    from memos.memories.textual.item import TextualMemoryItem

logger = get_logger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


class MMRReranker(BaseReranker):
    """
    MMR (Maximal Marginal Relevance) Reranker for deduplication.

    Balances relevance and diversity using the formula:
    score(i) = λ * rel(i) - (1-λ) * red(i) - α * tag(i)

    Where:
    - rel(i): relevance to query (cosine similarity)
    - red(i): redundancy with selected items (max similarity)
    - tag(i): tag overlap with selected items (Jaccard similarity)
    """

    def __init__(
        self,
        lambda_param: float = 0.75,
        alpha: float = 0.15,
        tag_threshold: float = 0.5,
        level_weights: dict[str, float] | None = None,
        level_field: str = "background",
        **kwargs,
    ):
        """
        Initialize MMR Reranker.

        Args:
            lambda_param: Weight for relevance vs diversity (0-1), default 0.7
            alpha: Weight for tag penalty (>=0), default 0.3
            tag_threshold: Threshold for tag similarity (not used in current formula)
            level_weights: Optional weights for different memory levels
            level_field: Field name for memory level in metadata
        """
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.tag_threshold = tag_threshold
        self.level_weights = level_weights or {"topic": 1.0, "concept": 1.0, "fact": 1.0}
        self.level_field = level_field

    @timed
    def rerank(
        self,
        query: str,
        graph_results: list[TextualMemoryItem],
        top_k: int,
        search_filter: dict | None = None,
        **kwargs,
    ) -> list[tuple[TextualMemoryItem, float]]:
        """
        Rerank using MMR algorithm.

        Args:
            query: Query string
            graph_results: List of candidate memory items
            top_k: Number of top results to return
            search_filter: Optional filter (not used in MMR)
            **kwargs: Additional arguments, may include:
                - 'query_embedding': Query embedding vector
                - If not provided, will use existing relativity scores

        Returns:
            List of (item, score) tuples sorted by MMR score
        """
        if not graph_results:
            return []

        # Extract embeddings from candidates
        items_with_emb, embeddings = self._extract_embeddings(graph_results)

        if not items_with_emb:
            logger.warning("[MMR] No items with embeddings, returning original order")
            return [(item, 0.5) for item in graph_results[:top_k]]

        # Get or calculate relevance scores
        query_embedding: list[float] | None = kwargs.get("query_embedding")

        if query_embedding:
            # Calculate relevance scores from query embedding
            rel_scores = self._cosine_similarities(query_embedding, embeddings)
            # Apply level weights if configured
            rel_scores = self._apply_level_weights(items_with_emb, rel_scores)
        else:
            # Use existing relativity scores from metadata
            rel_scores = self._extract_relativity_scores(items_with_emb)
            if not rel_scores:
                logger.warning(
                    "[MMR] No query_embedding and no relativity scores, returning original order"
                )
                return [(item, 0.5) for item in graph_results[:top_k]]

        # Extract tags for all items
        all_tags = [self._extract_tags(item) for item in items_with_emb]

        # MMR iterative selection
        selected_indices = []
        unselected_indices = list(range(len(items_with_emb)))

        for _ in range(min(top_k, len(items_with_emb))):
            best_idx = None
            best_score = float("-inf")

            for idx in unselected_indices:
                # Calculate redundancy with selected items
                if selected_indices:
                    redundancy = max(
                        self._cosine_similarity(embeddings[idx], embeddings[j])
                        for j in selected_indices
                    )
                else:
                    redundancy = 0.0

                # Calculate tag overlap with selected items
                if selected_indices and self.alpha > 0:
                    tag_penalty = max(
                        self._jaccard_similarity(all_tags[idx], all_tags[j])
                        for j in selected_indices
                    )
                else:
                    tag_penalty = 0.0

                # MMR score
                mmr_score = (
                    self.lambda_param * rel_scores[idx]
                    - (1 - self.lambda_param) * redundancy
                    - self.alpha * tag_penalty
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                unselected_indices.remove(best_idx)

        # Build result with original relevance scores
        result = []
        for idx in selected_indices:
            item = items_with_emb[idx]
            # Use original relevance score for the returned score
            result.append((item, rel_scores[idx]))

        logger.info(
            f"[MMR] Reranked {len(graph_results)} items, "
            f"selected {len(result)} (lambda={self.lambda_param}, alpha={self.alpha})"
        )

        return result

    def _extract_embeddings(
        self, items: list[TextualMemoryItem]
    ) -> tuple[list[TextualMemoryItem], list[list[float]]]:
        """Extract items that have embeddings and their embedding vectors."""
        items_with_emb = []
        embeddings = []

        for item in items:
            emb = self._get_embedding(item)
            if emb:
                items_with_emb.append(item)
                embeddings.append(emb)

        return items_with_emb, embeddings

    def _get_embedding(self, item: Any) -> list[float] | None:
        """Get embedding from item (handles both dict and object formats)."""
        if isinstance(item, dict):
            return item.get("metadata", {}).get("embedding")
        elif hasattr(item, "metadata"):
            return getattr(item.metadata, "embedding", None)
        return None

    def _extract_relativity_scores(self, items: list[TextualMemoryItem]) -> list[float] | None:
        """
        Extract existing relativity scores from item metadata.

        These scores are typically calculated during retrieval as cosine similarity
        between query and item embeddings, so we can reuse them instead of recalculating.

        Args:
            items: List of memory items

        Returns:
            List of relativity scores, or None if any item lacks a score
        """
        scores = []
        for item in items:
            score = None
            if isinstance(item, dict):
                score = item.get("metadata", {}).get("relativity")
            elif hasattr(item, "metadata"):
                score = getattr(item.metadata, "relativity", None) or item.metadata.get("model_extra").get("relativity")

            if score is None:
                # If any item lacks a relativity score, we can't use this optimization
                return None
            scores.append(float(score))

        return scores

    def _extract_tags(self, item: Any) -> set[str]:
        """Extract tags from item metadata."""
        tags = None

        if isinstance(item, dict):
            metadata = item.get("metadata", {})
            tags = metadata.get("tags") or metadata.get("tag") or metadata.get("labels")
        elif hasattr(item, "metadata"):
            tags = (
                getattr(item.metadata, "tags", None)
                or getattr(item.metadata, "tag", None)
                or getattr(item.metadata, "labels", None)
            )

        if tags is None:
            return set()

        # Convert to set
        if isinstance(tags, str):
            return {tags}
        elif isinstance(tags, (list, tuple)):
            return set(tags)
        elif isinstance(tags, set):
            return tags
        else:
            return set()

    def _apply_level_weights(
        self, items: list[TextualMemoryItem], scores: list[float]
    ) -> list[float]:
        """Apply level weights to relevance scores."""
        weighted_scores = []
        for item, score in zip(items, scores, strict=False):
            weight = self._get_level_weight(item)
            weighted_scores.append(score * weight)
        return weighted_scores

    def _get_level_weight(self, item: Any) -> float:
        """Get level weight for an item."""
        level = None
        if isinstance(item, dict):
            level = item.get("metadata", {}).get(self.level_field)
        elif hasattr(item, "metadata"):
            level = getattr(item.metadata, self.level_field, None)

        return self.level_weights.get(level, 1.0) if level else 1.0

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if _HAS_NUMPY:
            v1 = np.array(vec1, dtype=float)
            v2 = np.array(vec2, dtype=float)

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(v1, v2) / (norm1 * norm2))
        else:
            # Pure Python implementation
            dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

    def _cosine_similarities(
        self, query_vec: list[float], candidate_vecs: list[list[float]]
    ) -> list[float]:
        """Calculate cosine similarities between query and multiple candidates."""
        if _HAS_NUMPY:
            q = np.array(query_vec, dtype=float)
            candidates = np.array(candidate_vecs, dtype=float)

            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                return [0.0] * len(candidate_vecs)

            c_norms = np.linalg.norm(candidates, axis=1)
            dots = candidates @ q

            # Avoid division by zero
            similarities = np.zeros(len(candidate_vecs))
            mask = c_norms != 0
            similarities[mask] = dots[mask] / (c_norms[mask] * q_norm)

            return similarities.tolist()
        else:
            # Pure Python implementation
            return [self._cosine_similarity(query_vec, vec) for vec in candidate_vecs]

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union
