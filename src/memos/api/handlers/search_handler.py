"""
Search handler for memory search functionality (Class-based version).

This module provides a class-based implementation of search handlers,
using dependency injection for better modularity and testability.
"""

from typing import Any

from memos.api.handlers.base_handler import BaseHandler, HandlerDependencies
from memos.api.product_models import APISearchRequest, SearchResponse
from memos.log import get_logger
from memos.memories.textual.tree_text_memory.retrieve.retrieve_utils import (
    cosine_similarity_matrix,
)
from memos.multi_mem_cube.composite_cube import CompositeCubeView
from memos.multi_mem_cube.single_cube import SingleCubeView
from memos.multi_mem_cube.views import MemCubeView


logger = get_logger(__name__)


class SearchHandler(BaseHandler):
    """
    Handler for memory search operations.

    Provides fast, fine-grained, and mixture-based search modes.
    """

    def __init__(self, dependencies: HandlerDependencies):
        """
        Initialize search handler.

        Args:
            dependencies: HandlerDependencies instance
        """
        super().__init__(dependencies)
        self._validate_dependencies(
            "naive_mem_cube", "mem_scheduler", "searcher", "deepsearch_agent"
        )

    def handle_search_memories(self, search_req: APISearchRequest) -> SearchResponse:
        """
        Main handler for search memories endpoint.

        Orchestrates the search process based on the requested search mode,
        supporting both text and preference memory searches.

        Args:
            search_req: Search request containing query and parameters

        Returns:
            SearchResponse with formatted results
        """
        self.logger.info(f"[SearchHandler] Search Req is: {search_req}")

        # Increase recall pool if deduplication is enabled to ensure diversity
        original_top_k = search_req.top_k
        if search_req.dedup == "sim":
            search_req.top_k = original_top_k * 5
        elif search_req.dedup == "mmr":
            # For MMR, also increase recall pool
            search_req.top_k = original_top_k * 3

        cube_view = self._build_cube_view(search_req)

        results = cube_view.search_memories(search_req)
        if search_req.dedup == "sim":
            results = self._dedup_text_memories(results, original_top_k)
            self._strip_embeddings(results)
            # Restore original top_k for downstream logic or response metadata
            search_req.top_k = original_top_k

        # Unified MMR deduplication
        elif search_req.dedup == "mmr" and self.reranker:
            self.logger.info(
                f"[SearchHandler] Using unified deduplication with {type(self.reranker).__name__}"
            )
            text_mem_deduped, pref_mem_deduped = self._unified_mmr_dedup(
                results=results,
                query=search_req.query,
                original_top_k=original_top_k,
                pref_top_k=search_req.pref_top_k,
            )
            results["text_mem"] = text_mem_deduped
            results["pref_mem"] = pref_mem_deduped
            self._strip_embeddings(results)
            search_req.top_k = original_top_k

        self.logger.info(
            f"[SearchHandler] Final search results: count={len(results)} results={results}"
        )

        return SearchResponse(
            message="Search completed successfully",
            data=results,
        )

    def _dedup_text_memories(self, results: dict[str, Any], target_top_k: int) -> dict[str, Any]:
        buckets = results.get("text_mem", [])
        if not buckets:
            return results

        flat: list[tuple[int, dict[str, Any], float]] = []
        for bucket_idx, bucket in enumerate(buckets):
            for mem in bucket.get("memories", []):
                score = mem.get("metadata", {}).get("relativity", 0.0)
                flat.append((bucket_idx, mem, score))

        if len(flat) <= 1:
            return results

        embeddings = self._extract_embeddings([mem for _, mem, _ in flat])
        if embeddings is None:
            documents = [mem.get("memory", "") for _, mem, _ in flat]
            embeddings = self.searcher.embedder.embed(documents)

        similarity_matrix = cosine_similarity_matrix(embeddings)

        indices_by_bucket: dict[int, list[int]] = {i: [] for i in range(len(buckets))}
        for flat_index, (bucket_idx, _, _) in enumerate(flat):
            indices_by_bucket[bucket_idx].append(flat_index)

        selected_global: list[int] = []
        selected_by_bucket: dict[int, list[int]] = {i: [] for i in range(len(buckets))}

        ordered_indices = sorted(range(len(flat)), key=lambda idx: flat[idx][2], reverse=True)
        for idx in ordered_indices:
            bucket_idx = flat[idx][0]
            if len(selected_by_bucket[bucket_idx]) >= target_top_k:
                continue
            # Use 0.92 threshold strictly
            if self._is_unrelated(idx, selected_global, similarity_matrix, 0.92):
                selected_by_bucket[bucket_idx].append(idx)
                selected_global.append(idx)

        # Removed the 'filling' logic that was pulling back similar items.
        # Now it will only return items that truly pass the 0.92 threshold,
        # up to target_top_k.

        for bucket_idx, bucket in enumerate(buckets):
            selected_indices = selected_by_bucket.get(bucket_idx, [])
            bucket["memories"] = [flat[i][1] for i in selected_indices]
        return results

    @staticmethod
    def _is_unrelated(
        index: int,
        selected_indices: list[int],
        similarity_matrix: list[list[float]],
        similarity_threshold: float,
    ) -> bool:
        return all(similarity_matrix[index][j] <= similarity_threshold for j in selected_indices)

    @staticmethod
    def _max_similarity(
        index: int, selected_indices: list[int], similarity_matrix: list[list[float]]
    ) -> float:
        if not selected_indices:
            return 0.0
        return max(similarity_matrix[index][j] for j in selected_indices)

    @staticmethod
    def _extract_embeddings(memories: list[dict[str, Any]]) -> list[list[float]] | None:
        embeddings: list[list[float]] = []
        for mem in memories:
            embedding = mem.get("metadata", {}).get("embedding")
            if not embedding:
                return None
            embeddings.append(embedding)
        return embeddings

    @staticmethod
    def _strip_embeddings(results: dict[str, Any]) -> None:
        for bucket in results.get("text_mem", []):
            for mem in bucket.get("memories", []):
                metadata = mem.get("metadata", {})
                if "embedding" in metadata:
                    metadata["embedding"] = []
        for bucket in results.get("tool_mem", []):
            for mem in bucket.get("memories", []):
                metadata = mem.get("metadata", {})
                if "embedding" in metadata:
                    metadata["embedding"] = []

    def _resolve_cube_ids(self, search_req: APISearchRequest) -> list[str]:
        """
        Normalize target cube ids from search_req.
        Priority:
        1) readable_cube_ids (deprecated mem_cube_id is converted to this in model validator)
        2) fallback to user_id
        """
        if search_req.readable_cube_ids:
            return list(dict.fromkeys(search_req.readable_cube_ids))

        return [search_req.user_id]

    def _build_cube_view(self, search_req: APISearchRequest) -> MemCubeView:
        cube_ids = self._resolve_cube_ids(search_req)

        if len(cube_ids) == 1:
            cube_id = cube_ids[0]
            return SingleCubeView(
                cube_id=cube_id,
                naive_mem_cube=self.naive_mem_cube,
                mem_reader=self.mem_reader,
                mem_scheduler=self.mem_scheduler,
                logger=self.logger,
                searcher=self.searcher,
                deepsearch_agent=self.deepsearch_agent,
            )
        else:
            single_views = [
                SingleCubeView(
                    cube_id=cube_id,
                    naive_mem_cube=self.naive_mem_cube,
                    mem_reader=self.mem_reader,
                    mem_scheduler=self.mem_scheduler,
                    logger=self.logger,
                    searcher=self.searcher,
                    deepsearch_agent=self.deepsearch_agent,
                )
                for cube_id in cube_ids
            ]
            return CompositeCubeView(cube_views=single_views, logger=self.logger)

    def _unified_mmr_dedup(
        self,
        results: dict[str, Any],
        query: str,
        original_top_k: int,
        pref_top_k: int,
    ) -> tuple[list[dict], list[dict]]:
        """
        Unified deduplication for both text and preference memories.

        Strategy (same as Searcher):
        1. Use http_bge reranker to get accurate relevance_score for each item
        2. Use MMRReranker for deduplication with http_bge scores + embeddings

        This combines:
        - Accurate relevance scores from reranker service
        - Embedding-based redundancy calculation for diversity

        Args:
            results: Dictionary containing text_mem and pref_mem
            query: Search query
            original_top_k: Number of text memories to return
            pref_top_k: Number of preference memories to return

        Returns:
            Tuple of (text_mem_buckets, pref_mem_list)
        """
        # Extract text memories from buckets
        text_buckets = results.get("text_mem", [])
        all_text_memories = []
        for bucket in text_buckets:
            all_text_memories.extend(bucket.get("memories", []))

        # Extract preference memories
        pref_memories = results.get("pref_mem", [])

        self.logger.info(
            f"[Unified Dedup] Before: {len(all_text_memories)} text mems, "
            f"{len(pref_memories)} pref mems"
        )

        # Check if we have any memories to process
        if not all_text_memories and not pref_memories:
            return (results.get("text_mem", []), [])

        # Mark memory types for later separation
        for mem in all_text_memories:
            mem["_is_preference"] = False
        for mem in pref_memories:
            mem["_is_preference"] = True

        # Merge all memories
        all_memories = all_text_memories + pref_memories

        total_top_k = original_top_k + pref_top_k

        # Step 1: Use http_bge reranker to get accurate relevance scores
        try:
            self.logger.info(f"[Unified Dedup] Step 1: Getting relevance scores from http_bge reranker")
            reranked_with_scores = self.reranker.rerank(
                query=query,
                graph_results=all_memories,
                top_k=len(all_memories),  # Get scores for all items
            )

            self.logger.info(f"[Unified Dedup] Got {len(reranked_with_scores)} relevance scores from reranker")

            # Update items with accurate relevance scores from reranker
            items_with_relevance = []
            for item, reranker_score in reranked_with_scores:
                # Update score in metadata
                if isinstance(item, dict):
                    if "metadata" not in item:
                        item["metadata"] = {}
                    item["metadata"]["relativity"] = reranker_score
                else:
                    if not hasattr(item, "metadata"):
                        from types import SimpleNamespace
                        item.metadata = SimpleNamespace()
                    item.metadata.relativity = reranker_score
                items_with_relevance.append(item)

        except Exception as e:
            self.logger.error(f"[Unified Dedup] Reranker failed to get relevance scores: {e}", exc_info=True)
            return (text_buckets, pref_memories)

        # Step 2: Fetch embeddings if missing (for MMR redundancy calculation)
        for item in items_with_relevance:
            if isinstance(item, dict):
                embedding = item.get("metadata", {}).get("embedding")
                if not embedding or len(embedding) == 0:
                    # Try to compute embedding using embedder if available
                    if self.searcher and self.searcher.embedder:
                        try:
                            memory_text = item.get("memory", "")
                            if memory_text:
                                item["metadata"]["embedding"] = self.searcher.embedder.embed([memory_text])[0]
                                self.logger.debug(f"[Unified Dedup] Computed embedding for item")
                        except Exception as e:
                            self.logger.warning(f"[Unified Dedup] Failed to compute embedding: {e}")
            else:
                # Handle object-style items
                if not hasattr(item.metadata, "embedding") or not item.metadata.embedding or len(item.metadata.embedding) == 0:
                    if self.searcher and self.searcher.embedder:
                        try:
                            memory_text = getattr(item, "memory", "")
                            if memory_text:
                                item.metadata.embedding = self.searcher.embedder.embed([memory_text])[0]
                                self.logger.debug(f"[Unified Dedup] Computed embedding for item")
                        except Exception as e:
                            self.logger.warning(f"[Unified Dedup] Failed to compute embedding: {e}")

        # Step 3: Use MMR with reranker relevance scores and embeddings
        try:
            from memos.reranker.mmr import MMRReranker

            self.logger.info("[Unified Dedup] Step 2: Using MMR for deduplication")

            # Create MMRReranker with configured parameters
            mmr_reranker = MMRReranker(
                lambda_param=0.8,   # Balance between relevance and diversity
                alpha=0.15,         # Tag penalty weight
            )

            # Call MMR reranker
            # Note: We don't pass query_embedding, so MMR will use existing relativity scores
            # from metadata (which we just updated with accurate reranker scores)
            deduped_items = mmr_reranker.rerank(
                query=query,
                graph_results=items_with_relevance,
                top_k=total_top_k,
            )

            self.logger.info(f"[Unified Dedup] After MMR: {len(deduped_items)} results")

        except Exception as e:
            self.logger.error(f"[Unified Dedup] MMR deduplication failed: {e}", exc_info=True)
            # Fallback: use items with relevance scores but without MMR
            deduped_items = [(item, item.get("metadata", {}).get("relativity", 0.5) if isinstance(item, dict) else getattr(item.metadata, "relativity", 0.5)) for item in items_with_relevance[:total_top_k]]

        # Separate back to text and preference memories
        text_results = []
        pref_results = []

        for item, score in deduped_items:
            # Update score in metadata
            if isinstance(item, dict):
                if "metadata" not in item:
                    item["metadata"] = {}
                item["metadata"]["relativity"] = score
                is_pref = item.pop("_is_preference", False)
            else:
                if not hasattr(item, "metadata"):
                    from types import SimpleNamespace
                    item.metadata = SimpleNamespace()
                item.metadata.relativity = score
                is_pref = getattr(item, "_is_preference", False)
                if hasattr(item, "_is_preference"):
                    delattr(item, "_is_preference")

            if is_pref:
                pref_results.append(item)
            else:
                text_results.append(item)

        # Limit results
        text_results = text_results[:original_top_k]
        pref_results = pref_results[:pref_top_k]

        self.logger.info(
            f"[Unified Dedup] After: {len(text_results)} text mems, {len(pref_results)} pref mems"
        )

        # Reconstruct text_mem bucket structure
        new_text_mem = []
        if text_results:
            # Simplified: put all in one bucket
            new_text_mem = [
                {
                    "type": "mixed",
                    "memories": text_results,
                }
            ]

        return (new_text_mem, pref_results)
