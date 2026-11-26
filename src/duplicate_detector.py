"""Duplicate, near-duplicate, and suspicious repeated-phrase detection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DEFAULT_SIMILARITY_THRESHOLD
from src.nlp_preprocessing import clean_text, tokenize_without_stopwords


@dataclass
class DuplicateAnalysisResult:
    exact_groups: pd.DataFrame
    near_duplicate_pairs: pd.DataFrame
    repeated_phrases: pd.DataFrame
    template_clusters: pd.DataFrame
    summary: dict[str, Any]


def _union_find_clusters(num_items: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(num_items))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(num_items):
        groups[find(idx)].append(idx)

    return [sorted(members) for members in groups.values() if len(members) >= 2]


def analyze_duplicates_and_patterns(
    reviews: list[str],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_phrase_occurrences: int = 2,
) -> DuplicateAnalysisResult:
    """Detect exact duplicates, near-duplicates via TF-IDF cosine similarity, and repeated phrases."""
    raw_reviews = ["" if r is None else str(r).strip() for r in reviews]
    non_empty_indices = [i for i, r in enumerate(raw_reviews) if r]
    cleaned_all = [clean_text(r) for r in raw_reviews]

    # 1. Exact / normalized duplicates
    norm_map: dict[str, list[int]] = defaultdict(list)
    for idx in non_empty_indices:
        norm = cleaned_all[idx]
        if norm:
            norm_map[norm].append(idx)

    exact_rows = []
    exact_duplicate_review_count = 0
    for norm_text, idxs in norm_map.items():
        if len(idxs) >= 2:
            exact_duplicate_review_count += len(idxs)
            exact_rows.append(
                {
                    "occurrences": len(idxs),
                    "row_indices": ", ".join(str(i + 1) for i in idxs),
                    "sample_review": raw_reviews[idxs[0]],
                    "normalized_text": norm_text,
                }
            )
    exact_groups = (
        pd.DataFrame(exact_rows).sort_values("occurrences", ascending=False).reset_index(drop=True)
        if exact_rows
        else pd.DataFrame(columns=["occurrences", "row_indices", "sample_review", "normalized_text"])
    )

    # 2. Near-duplicate pairs & similarity clusters (TF-IDF word + char n-grams)
    near_pairs: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    near_dup_indices: set[int] = set()

    if len(non_empty_indices) >= 2:
        active_cleaned = [cleaned_all[i] for i in non_empty_indices]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        try:
            tfidf_matrix = vectorizer.fit_transform(active_cleaned)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            sim_matrix = np.zeros((len(active_cleaned), len(active_cleaned)))

        edges: list[tuple[int, int]] = []
        n = len(non_empty_indices)
        for a in range(n):
            for b in range(a + 1, n):
                sim = float(sim_matrix[a, b])
                if sim >= similarity_threshold:
                    idx_a = non_empty_indices[a]
                    idx_b = non_empty_indices[b]
                    is_exact = cleaned_all[idx_a] == cleaned_all[idx_b]
                    edges.append((a, b))
                    near_dup_indices.add(idx_a)
                    near_dup_indices.add(idx_b)
                    near_pairs.append(
                        {
                            "row_a": idx_a + 1,
                            "row_b": idx_b + 1,
                            "similarity_pct": round(sim * 100.0, 1),
                            "match_type": "Exact Match" if is_exact else "Near-Duplicate",
                            "review_a": raw_reviews[idx_a],
                            "review_b": raw_reviews[idx_b],
                        }
                    )

        clusters = _union_find_clusters(n, edges)
        for cluster_num, member_positions in enumerate(clusters, start=1):
            orig_indices = [non_empty_indices[pos] for pos in member_positions]
            pair_sims = [
                float(sim_matrix[member_positions[i], member_positions[j]])
                for i in range(len(member_positions))
                for j in range(i + 1, len(member_positions))
            ]
            avg_sim = float(np.mean(pair_sims)) * 100.0 if pair_sims else 100.0
            cluster_rows.append(
                {
                    "cluster_id": f"Cluster #{cluster_num}",
                    "review_count": len(orig_indices),
                    "avg_similarity_pct": round(avg_sim, 1),
                    "row_indices": ", ".join(str(i + 1) for i in orig_indices),
                    "representative_review": raw_reviews[orig_indices[0]],
                }
            )

    near_duplicate_pairs = (
        pd.DataFrame(near_pairs)
        .sort_values(["similarity_pct", "row_a"], ascending=[False, True])
        .reset_index(drop=True)
        if near_pairs
        else pd.DataFrame(columns=["row_a", "row_b", "similarity_pct", "match_type", "review_a", "review_b"])
    )

    template_clusters = (
        pd.DataFrame(cluster_rows)
        .sort_values("review_count", ascending=False)
        .reset_index(drop=True)
        if cluster_rows
        else pd.DataFrame(columns=["cluster_id", "review_count", "avg_similarity_pct", "row_indices", "representative_review"])
    )

    # 3. Repeated suspicious n-gram phrases across distinct reviews
    phrase_review_map: dict[str, set[int]] = defaultdict(set)
    phrase_total_counts: Counter[str] = Counter()

    for idx in non_empty_indices:
        tokens = tokenize_without_stopwords(raw_reviews[idx])
        seen_in_review: set[str] = set()
        for n_gram_len in (3, 4):
            if len(tokens) < n_gram_len:
                continue
            for i in range(len(tokens) - n_gram_len + 1):
                phrase = " ".join(tokens[i : i + n_gram_len])
                phrase_total_counts[phrase] += 1
                if phrase not in seen_in_review:
                    phrase_review_map[phrase].add(idx)
                    seen_in_review.add(phrase)

    phrase_rows = []
    for phrase, review_set in phrase_review_map.items():
        if len(review_set) >= min_phrase_occurrences:
            phrase_rows.append(
                {
                    "phrase": phrase,
                    "reviews_containing": len(review_set),
                    "total_occurrences": phrase_total_counts[phrase],
                    "word_length": len(phrase.split()),
                    "row_indices": ", ".join(str(i + 1) for i in sorted(review_set)[:12]),
                }
            )

    repeated_phrases = (
        pd.DataFrame(phrase_rows)
        .sort_values(["reviews_containing", "word_length", "total_occurrences"], ascending=[False, False, False])
        .head(25)
        .reset_index(drop=True)
        if phrase_rows
        else pd.DataFrame(columns=["phrase", "reviews_containing", "total_occurrences", "word_length", "row_indices"])
    )

    summary = {
        "total_reviews_checked": len(non_empty_indices),
        "exact_duplicate_groups": int(len(exact_groups)),
        "exact_duplicate_reviews": int(exact_duplicate_review_count),
        "near_duplicate_pairs": int(len(near_duplicate_pairs)),
        "reviews_in_similarity_clusters": int(len(near_dup_indices)),
        "similarity_clusters_count": int(len(template_clusters)),
        "repeated_phrases_count": int(len(repeated_phrases)),
    }

    return DuplicateAnalysisResult(
        exact_groups=exact_groups,
        near_duplicate_pairs=near_duplicate_pairs,
        repeated_phrases=repeated_phrases,
        template_clusters=template_clusters,
        summary=summary,
    )
