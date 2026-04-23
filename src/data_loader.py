"""
data_loader.py
--------------
Loads TriviaQA from HuggingFace, builds a shared retrieval corpus,
and splits queries into train / test sets.

Corpus design: Every query's Wikipedia evidence pages go into a shared pool,
so retrieval is non-trivial — each query must find its own documents among
all others' documents acting as distractors.
"""

import logging
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)


class TriviaQALoader:
    """Loads TriviaQA (rc.wikipedia) and constructs a retrieval corpus."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["dataset"]

    # ------------------------------------------------------------------
    def load(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Returns
        -------
        corpus       : list of {doc_id, title, text}
        train_queries: list of {query_id, question, relevant_doc_ids, answers}
        test_queries : list of {query_id, question, relevant_doc_ids, answers}
        """
        logger.info(
            "Loading TriviaQA  config=%s  split=%s",
            self.cfg["config"],
            self.cfg["split"],
        )
        dataset = load_dataset(
            self.cfg["name"],
            self.cfg["config"],
            split=self.cfg["split"],
            trust_remote_code=True,
        )

        random.seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

        total = min(self.cfg["num_queries"], len(dataset))
        indices = random.sample(range(len(dataset)), total)

        corpus: Dict[str, Dict] = {}   # doc_id → doc
        queries: List[Dict] = []

        for idx in indices:
            sample = dataset[idx]
            qid = sample["question_id"]
            question = sample["question"]

            # Collect all answer aliases for EM / F1 evaluation
            answers = list(
                set(sample["answer"]["aliases"] + [sample["answer"]["value"]])
            )

            relevant_doc_ids: List[str] = []

            # Extract Wikipedia evidence pages
            entity_pages = sample.get("entity_pages", {})
            wiki_contexts = entity_pages.get("wiki_context", []) or []
            titles = entity_pages.get("title", []) or []

            for i, (title, context) in enumerate(zip(titles, wiki_contexts)):
                if not (context and context.strip()):
                    continue
                doc_id = f"{qid}__wiki_{i}"
                if doc_id not in corpus:
                    corpus[doc_id] = {
                        "doc_id": doc_id,
                        "title": title,
                        "text": context[:2000],  # truncate for memory efficiency
                    }
                relevant_doc_ids.append(doc_id)

            # Only keep queries that actually have evidence
            if relevant_doc_ids:
                queries.append(
                    {
                        "query_id": qid,
                        "question": question,
                        "relevant_doc_ids": relevant_doc_ids,
                        "answers": answers,
                    }
                )

        corpus_list = list(corpus.values())
        n_train = self.cfg["num_train_queries"]
        n_test = self.cfg["num_test_queries"]

        # Shuffle queries before splitting so train/test are representative
        random.shuffle(queries)
        train_queries = queries[:n_train]
        test_queries = queries[n_train : n_train + n_test]

        logger.info(
            "Corpus: %d docs | Train: %d queries | Test: %d queries",
            len(corpus_list),
            len(train_queries),
            len(test_queries),
        )
        return corpus_list, train_queries, test_queries
