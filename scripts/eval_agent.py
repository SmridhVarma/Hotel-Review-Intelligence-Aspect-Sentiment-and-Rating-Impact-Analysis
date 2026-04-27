"""
Module C Evaluation Script
Usage: python scripts/eval_agent.py

Runs 20 queries through the live agent, computes automatic metrics,
then uses GPT-4o as a judge for relevance and context accuracy.
Results saved to outputs/agent_eval_results.csv and printed to stdout.
"""

import json
import os
import re
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC  = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(ROOT, ".env"))

from agent.graph import run_query
from paths import OUTPUT_DIR

# ── Eval query set ────────────────────────────────────────────────────────────

EVAL_QUERIES = [
    {"query": "What do guests complain about regarding cleanliness?",        "hotel": "__global__", "type": "evidence"},
    {"query": "What aspects of the staff do guests praise most?",            "hotel": "__global__", "type": "evidence"},
    {"query": "What noise problems do guests mention?",                      "hotel": "__global__", "type": "evidence"},
    {"query": "What do guests say about the location?",                      "hotel": "__global__", "type": "evidence"},
    {"query": "What complaints do guests have about the rooms?",             "hotel": "__global__", "type": "evidence"},
    {"query": "Which aspect should this hotel prioritise improving?",        "hotel": "41",         "type": "prioritisation"},
    {"query": "What is dragging down ratings the most globally?",            "hotel": "__global__", "type": "prioritisation"},
    {"query": "Which aspect has the biggest positive impact on guest scores?","hotel": "__global__", "type": "prioritisation"},
    {"query": "What should Hotel Arena focus on to improve its rating?",     "hotel": "Hotel Arena","type": "prioritisation"},
    {"query": "Are there guests who gave positive reviews but low scores?",  "hotel": "__global__", "type": "mismatch"},
    {"query": "Which aspects show the biggest mismatch between text and rating?","hotel": "__global__","type": "mismatch"},
    {"query": "What do business travellers complain about most?",            "hotel": "__global__", "type": "segment"},
    {"query": "What do families say about the rooms?",                       "hotel": "__global__", "type": "segment"},
    {"query": "What aspects do couples rate most positively?",               "hotel": "__global__", "type": "segment"},
    {"query": "What do solo travellers complain about regarding noise?",     "hotel": "__global__", "type": "segment"},
    {"query": "What do guests say about the infinity pool and spa facilities?","hotel": "__global__","type": "evidence"},
    {"query": "Tell me about Hotel Xyzabc Nonexistent 1234",                 "hotel": "Hotel Xyzabc Nonexistent 1234","type": "evidence"},
    {"query": "What is the price range for rooms?",                          "hotel": "__global__", "type": "evidence"},
    {"query": "How is the cleanliness at 11 Cadogan Gardens?",               "hotel": "11 Cadogan Gardens","type": "evidence"},
    {"query": "Which aspect matters most for business travellers at The Savoy?","hotel": "The Savoy","type": "segment"},
]

# ── Run queries ───────────────────────────────────────────────────────────────

def run_queries():
    results = []
    for i, q in enumerate(EVAL_QUERIES):
        print(f"[{i+1}/{len(EVAL_QUERIES)}] {q['type']:15s} | {q['query'][:60]}")
        try:
            result = run_query(
                query=q["query"],
                hotel_name=q["hotel"],
                thread_id=f"eval_{i}",
            )
            results.append({
                "query_type":      q["type"],
                "query":           q["query"],
                "hotel":           q["hotel"],
                "response":        result.get("response", ""),
                "citations":       result.get("citations", []),
                "low_confidence":  result.get("low_confidence", False),
                "hotel_unresolved":result.get("hotel_unresolved", False),
                "summary_context": result.get("summary_context"),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "query_type":      q["type"],
                "query":           q["query"],
                "hotel":           q["hotel"],
                "response":        "",
                "citations":       [],
                "low_confidence":  True,
                "hotel_unresolved":False,
                "summary_context": None,
                "error":           str(e),
            })
    return results

# ── Automatic metrics ─────────────────────────────────────────────────────────

def compute_auto_metrics(results):
    resolved    = [r for r in results if not r["low_confidence"]]
    resolution  = len(resolved) / len(results)
    grounded    = sum(1 for r in resolved if len(r["citations"]) > 0) / len(resolved) if resolved else 0
    all_sims    = [c.get("similarity_score", 0) for r in results for c in r["citations"]]
    mean_sim    = float(np.mean(all_sims)) if all_sims else float("nan")
    unresolved  = sum(1 for r in results if r["hotel_unresolved"]) / len(results)

    print("\n=== Automatic Metrics ===")
    print(f"  Resolution rate:            {resolution:.2%}")
    print(f"  Citation coverage:          {grounded:.2%}  (of resolved responses)")
    print(f"  Mean retrieval similarity:  {mean_sim:.4f}")
    print(f"  Hotel unresolved rate:      {unresolved:.2%}")
    print()
    print("  Resolution rate by query type:")
    for t in ["evidence", "prioritisation", "mismatch", "segment"]:
        sub = [r for r in results if r["query_type"] == t]
        rate = sum(1 for r in sub if not r["low_confidence"]) / len(sub) if sub else 0
        print(f"    {t:15s}: {rate:.0%}")

    return resolution, grounded, mean_sim

# ── GPT-4o judge ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an evaluation judge for a hotel analytics AI assistant.
Score the response on two dimensions. Return ONLY a JSON object with keys
"relevance" and "context_accuracy", each an integer 1-5.

Relevance (1-5): Does the response directly answer the user's question?
  5=perfectly on-topic  4=mostly  3=partial  2=tangential  1=off-topic or empty

Context accuracy (1-5): Does the response correctly use the retrieved evidence?
Evidence may be review sentences (cited) or a SHAP impact summary (narrative).
  5=all claims grounded  4=mostly  3=some unsupported  2=mostly ungrounded  1=ignores evidence

Question: {query}
Response: {response}
Retrieved context: {context}"""


def judge_response(client, query, response, citations, summary_context):
    if not response.strip():
        return {"relevance": 1, "context_accuracy": 1}

    if citations:
        ctx = f"{len(citations)} review sentence citations"
    elif summary_context:
        ctx = f"SHAP impact summary: {str(summary_context)[:300]}"
    else:
        ctx = "no retrieved context"

    prompt = JUDGE_PROMPT.format(
        query=query,
        response=response[:800],
        context=ctx,
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=60,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception:
        nums = re.findall(r"[1-5]", raw)
        return {
            "relevance":        int(nums[0]) if len(nums) > 0 else 3,
            "context_accuracy": int(nums[1]) if len(nums) > 1 else 3,
        }


def run_judge(results):
    client = OpenAI()
    print("\n=== GPT-4o Judge ===")
    for i, row in enumerate(results):
        scores = judge_response(
            client,
            row["query"],
            row["response"],
            row["citations"],
            row.get("summary_context"),
        )
        results[i]["relevance"]        = scores.get("relevance", 3)
        results[i]["context_accuracy"] = scores.get("context_accuracy", 3)
        print(f"  [{i+1:2d}/{len(results)}] R={results[i]['relevance']}  CA={results[i]['context_accuracy']}  {row['query'][:55]}")
    return results

# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results):
    resolved = [r for r in results if not r["low_confidence"]]
    all_sims = [c.get("similarity_score", 0) for r in results for c in r["citations"]]

    relevance  = [r["relevance"]        for r in results if "relevance"        in r]
    ctx_acc    = [r["context_accuracy"] for r in results if "context_accuracy" in r]
    grounded   = sum(1 for r in resolved if len(r["citations"]) > 0) / len(resolved) if resolved else 0

    print("\n=== Module C Evaluation Summary ===")
    print(f"  {'Groundedness (citation coverage)':45s}: {grounded:.2%}")
    print(f"  {'Relevance (GPT-4o judge, 1-5)':45s}: {np.mean(relevance):.2f} +/- {np.std(relevance):.2f}")
    print(f"  {'Context accuracy (GPT-4o, 1-5)':45s}: {np.mean(ctx_acc):.2f} +/- {np.std(ctx_acc):.2f}")
    print(f"  {'Resolution rate':45s}: {len(resolved)/len(results):.2%}")
    print(f"  {'Mean retrieval similarity':45s}: {np.mean(all_sims):.4f}" if all_sims else f"  {'Mean retrieval similarity':45s}: nan")

    print("\n  Per query type:")
    for t in ["evidence", "prioritisation", "mismatch", "segment"]:
        sub = [r for r in results if r["query_type"] == t]
        if not sub:
            continue
        res  = sum(1 for r in sub if not r["low_confidence"]) / len(sub)
        rel  = np.mean([r.get("relevance", 1) for r in sub])
        ca   = np.mean([r.get("context_accuracy", 1) for r in sub])
        print(f"    {t:15s}: resolution={res:.0%}  relevance={rel:.1f}  context_acc={ca:.1f}")

# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(results):
    out_path = os.path.join(OUTPUT_DIR, "agent_eval_results.json")
    serialisable = []
    for r in results:
        row = dict(r)
        # summary_context can be large — keep it but truncate for readability
        if row.get("summary_context"):
            row["summary_context"] = row["summary_context"][:500]
        serialisable.append(row)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Module C evaluation...\n")
    results = run_queries()
    compute_auto_metrics(results)
    results = run_judge(results)
    print_summary(results)
    save_results(results)
