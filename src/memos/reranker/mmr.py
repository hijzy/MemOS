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
        alpha: float = 0.3,
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


class TwoStageMMRDeduplicator:
    """
    两阶段MMR去重器：统一的去重逻辑，避免代码重复。

    策略：
    1. 粗排（Coarse Ranking）：使用embedding MMR快速过滤，取top_k * coarse_factor
       - 使用数据库embedding计算相似度
       - 本地快速计算
    2. 精排（Fine Ranking）：使用配置的reranker精确排序，取top_k
       - 可以是http_bge或其他reranker
       - 提供准确的relevance分数

    使用场景：
    - search_handler: 对text_mem和pref_mem统一去重
    - searcher: 对检索结果去重

    参数全部可配置，适合不同场景。
    """

    def __init__(
        self,
        reranker: BaseReranker,
        graph_store: Any,  # Neo4jGraphDB
        embedder: Any,  # OllamaEmbedder
        lambda_param: float = 0.8,
        alpha: float = 0.1,
        coarse_factor: int = 3,
    ):
        """
        初始化两阶段MMR去重器。

        Args:
            reranker: 精排用的reranker（如HTTPBGEReranker）
            graph_store: 图数据库实例，用于获取embedding
            embedder: Embedder实例，用于计算query的embedding
            lambda_param: MMR的lambda参数（相关性权重）
            alpha: MMR的alpha参数（标签惩罚权重）
            coarse_factor: 粗排倍数，粗排取top_k * coarse_factor个
        """
        self.reranker = reranker
        self.graph_store = graph_store
        self.embedder = embedder
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.coarse_factor = coarse_factor

        # 创建MMR实例用于粗排
        self.mmr_reranker = MMRReranker(
            lambda_param=lambda_param,
            alpha=alpha,
        )

    @timed
    def deduplicate(
        self,
        query: str,
        candidates: list,  # TextualMemoryItem或dict
        top_k: int,
        query_embedding: list[float] | None = None,  # 新增：可选的query_embedding
    ) -> list[tuple]:
        """
        两阶段MMR去重的主方法。

        Args:
            query: 查询字符串
            candidates: 候选items列表（可以是TextualMemoryItem或dict）
            top_k: 最终返回的数量
            query_embedding: 可选的query embedding（如searcher中已计算的cot_embedding）
                           如果提供，直接使用；否则用embedder现场计算

        Returns:
            List of (item, score) tuples，按MMR分数排序
        """
        if not candidates:
            return []

        logger.info(
            f"[TwoStage MMR] Starting with {len(candidates)} candidates, "
            f"target top_k={top_k}, coarse_factor={self.coarse_factor}"
        )

        # ===== Stage 1: 粗排 - 使用embedding MMR =====
        coarse_top_k = min(top_k * self.coarse_factor, len(candidates))

        logger.info(f"[TwoStage MMR] Stage 1: Coarse ranking with embedding MMR (top_k={coarse_top_k})")

        # Step 1.1: 获取或计算query的embedding
        if query_embedding is not None:
            logger.info(f"[TwoStage MMR] Using provided query_embedding (e.g., cot_embedding from searcher)")
        else:
            # Fallback: 如果没有提供query_embedding，用embedder计算
            try:
                query_embedding = self.embedder.embed([query])[0]
                logger.info(f"[TwoStage MMR] Computed query embedding with embedder, dimension: {len(query_embedding)}")
            except Exception as e:
                logger.error(f"[TwoStage MMR] Failed to compute query embedding: {e}", exc_info=True)
                # Fallback: 如果无法计算query embedding，直接用reranker
                logger.warning("[TwoStage MMR] Skipping coarse ranking, using reranker directly")
                try:
                    return self.reranker.rerank(
                        query=query,
                        graph_results=candidates,
                        top_k=top_k,
                    )
                except Exception as e2:
                    logger.error(f"[TwoStage MMR] Fallback reranker also failed: {e2}")
                    return [(item, self._get_relativity(item)) for item in candidates[:top_k]]

        # Step 1.2: 确保所有candidates有embedding（从数据库获取）
        self._ensure_embeddings(candidates)

        # Step 1.3: MMR粗排（传入query_embedding，现场计算相似度）
        try:
            coarse_results = self.mmr_reranker.rerank(
                query=query,
                graph_results=candidates,
                top_k=coarse_top_k,
                query_embedding=query_embedding,  # 使用提供的或计算的query_embedding
            )
            logger.info(f"[TwoStage MMR] Stage 1 done: {len(coarse_results)} items after coarse ranking")
        except Exception as e:
            logger.error(f"[TwoStage MMR] Coarse ranking failed: {e}", exc_info=True)
            # Fallback: 使用原始结果
            coarse_results = [(item, self._get_relativity(item)) for item in candidates[:coarse_top_k]]

        # ===== Stage 2: 精排 - 使用reranker =====
        logger.info(f"[TwoStage MMR] Stage 2: Fine ranking with reranker (top_k={top_k})")

        # 提取粗排后的items
        coarse_items = [item for item, score in coarse_results]

        try:
            # 使用配置的reranker精排
            fine_results = self.reranker.rerank(
                query=query,
                graph_results=coarse_items,
                top_k=top_k,
            )
            logger.info(f"[TwoStage MMR] Stage 2 done: {len(fine_results)} items after fine ranking")
        except Exception as e:
            logger.error(f"[TwoStage MMR] Fine ranking failed: {e}", exc_info=True)
            # Fallback: 使用粗排结果
            fine_results = coarse_results[:top_k]

        logger.info(
            f"[TwoStage MMR] Completed: returned {len(fine_results)} items "
            f"(lambda={self.lambda_param}, alpha={self.alpha})"
        )

        return fine_results

    def _ensure_embeddings(self, items: list) -> None:
        """
        确保所有items都有embedding，如果缺失则从数据库获取。

        Args:
            items: 候选items列表（会被就地修改）
        """
        for item in items:
            if isinstance(item, dict):
                # 处理dict格式
                embedding = item.get("metadata", {}).get("embedding")
                if not embedding or len(embedding) == 0:
                    self._fetch_embedding_dict(item)
            else:
                # 处理object格式 (TextualMemoryItem)
                if not hasattr(item, "metadata") or not hasattr(item.metadata, "embedding") or not item.metadata.embedding or len(item.metadata.embedding) == 0:
                    self._fetch_embedding_object(item)

    def _fetch_embedding_dict(self, item: dict) -> None:
        """从数据库获取embedding（dict格式）"""
        try:
            item_id = item.get("id")
            if item_id and self.graph_store:
                node = self.graph_store.get_node(item_id, include_embedding=True)
                if node and node.get("metadata", {}).get("embedding"):
                    if "metadata" not in item:
                        item["metadata"] = {}
                    item["metadata"]["embedding"] = node["metadata"]["embedding"]
                    logger.debug(f"[TwoStage MMR] Fetched embedding from DB for item {item_id}")
        except Exception as e:
            logger.warning(f"[TwoStage MMR] Failed to fetch embedding: {e}")

    def _fetch_embedding_object(self, item: Any) -> None:
        """从数据库获取embedding（object格式）"""
        try:
            item_id = getattr(item, "id", None)
            if item_id and self.graph_store:
                node = self.graph_store.get_node(item_id, include_embedding=True)
                if node and node.get("metadata", {}).get("embedding"):
                    if not hasattr(item, "metadata"):
                        from types import SimpleNamespace
                        item.metadata = SimpleNamespace()
                    item.metadata.embedding = node["metadata"]["embedding"]
                    logger.debug(f"[TwoStage MMR] Fetched embedding from DB for item {item_id}")
        except Exception as e:
            logger.warning(f"[TwoStage MMR] Failed to fetch embedding: {e}")

    def _get_relativity(self, item: Any) -> float:
        """获取item的relativity分数，用于fallback"""
        if isinstance(item, dict):
            return item.get("metadata", {}).get("relativity", 0.5)
        elif hasattr(item, "metadata"):
            return getattr(item.metadata, "relativity", 0.5)
        return 0.5

