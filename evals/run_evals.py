"""Offline evaluation of the RAG pipeline against a golden dataset.

    python -m evals.run_evals            # uses the adapters configured in .env
    python -m evals.run_evals --k 3

Two layers, measured separately (say this in interviews):
  - RETRIEVAL:  recall@k  -> is the expected source among the top-k chunks?
                MRR       -> how high is it ranked?
  - GENERATION: answer contains the key fact, answer is grounded (cites a chunk)

Run it before and after every change (chunking, embeddings, prompt, model)
and keep the numbers in the README. "Recall@4 went from X to Y" is the kind
of sentence that wins AI engineering interviews.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bodeguero.bootstrap import build_container
from bodeguero.config import get_settings
from bodeguero.domain.models import Domain

GOLDEN = Path(__file__).with_name("golden.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    container = build_container(get_settings())
    if container.store.count() == 0:
        container.ingestion().run()

    cases = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    hits = rr_sum = contains = grounded = 0

    print(f"{'#':>2}  {'recall':6}  {'fact':4}  {'grnd':4}  question")
    for i, case in enumerate(cases, 1):
        q_vec = container.embeddings.embed_query(case["question"])
        results = container.store.search(q_vec, args.k, domain=None)
        sources = [r.chunk.source for r in results]
        hit = case["expected_source"] in sources
        hits += hit
        rr_sum += 1 / (sources.index(case["expected_source"]) + 1) if hit else 0

        answer = container.answer_question.execute(case["question"])
        has_fact = case["must_contain"].lower() in answer.text.lower()
        contains += has_fact
        grounded += answer.grounded
        print(f"{i:>2}  {'yes' if hit else 'NO':6}  {'yes' if has_fact else 'NO':4}  "
              f"{'yes' if answer.grounded else 'NO':4}  {case['question']}")

    n = len(cases)
    print("\n--- results ---")
    print(f"recall@{args.k}:          {hits / n:.0%}")
    print(f"MRR:               {rr_sum / n:.2f}")
    print(f"answer has fact:   {contains / n:.0%}")
    print(f"grounded answers:  {grounded / n:.0%}")
    print(f"(llm={container.settings.llm_provider}, embeddings={container.settings.embeddings_provider}, "
          f"store={container.settings.vector_store})")

    # TODO(yee): add LLM-as-a-judge metrics once a real LLM is plugged in:
    #   - faithfulness: is every claim in the answer supported by the cited chunks?
    #   - answer relevance: does it actually answer the question?
    # Spot-check the judge by hand on a few cases before trusting it.
    _ = Domain  # domain-level metrics (routing accuracy) are a good next addition


if __name__ == "__main__":
    main()
