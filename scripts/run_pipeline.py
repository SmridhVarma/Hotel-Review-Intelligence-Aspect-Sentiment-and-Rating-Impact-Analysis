# =============================================================================
# run_pipeline.py — Full Pipeline Orchestrator
# =============================================================================
# Purpose:
#   Runs all pipeline stages end-to-end. Each stage is a separate import
#   and function call — stages can be skipped or re-run individually.
#
#   Stage 1  src.absa.preprocess          data/data.csv
#                                       → outputs/sentences.csv
#                                       → outputs/clean_reviews_stage1.csv
#   Stage 2  src.absa.aspect_extraction   outputs/sentences.csv
#                                       → outputs/aspect_dictionary.json
#   Stage 3  src.absa.sentiment_assignment outputs/sentences.csv +
#                                          outputs/aspect_dictionary.json
#                                       → outputs/aspect_sentences.csv
#                                       → outputs/review_features.csv
#   Stage 4  src.rating_impact.model      outputs/review_features.csv
#                                       → outputs/model_artifacts/
#                                       → outputs/shap_summary.json
#            src.rating_impact.evaluate   (evaluation report, same stage)
#                                       → outputs/evaluation_report.json
#   Stage 5  src.agent.ingest             outputs/aspect_sentences.csv +
#                                         outputs/shap_summary.json
#                                       → chromadb/
#                                       → outputs/hotel_names.json
#
# Usage:
#   python scripts/run_pipeline.py              # full pipeline (1 → 5)
#   python scripts/run_pipeline.py --from 3     # resume from stage 3
#   python scripts/run_pipeline.py --only 4     # run only stage 4
#   python scripts/run_pipeline.py --only 5     # re-ingest after fixes
#
# Input:  data/data.csv
# Output: all outputs/ artifacts + chromadb/
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure src/ is importable from anywhere
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
_SRC = os.path.join(_PROJECT_ROOT, "src")
for p in (_PROJECT_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}\n")


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_stage_1() -> None:
    _banner("Stage 1 — Text preprocessing")
    from absa.preprocess import run
    run()


def run_stage_2() -> None:
    _banner("Stage 2 — Aspect extraction (LDA)")
    from pathlib import Path
    from absa.aspect_extraction import run_fit, run_finalize

    sentences_csv = os.path.join(_PROJECT_ROOT, "outputs", "sentences.csv")
    cache_dir     = Path(os.path.join(_PROJECT_ROOT, "outputs", "_lda_cache"))
    output_path   = Path(os.path.join(_PROJECT_ROOT, "outputs", "aspect_dictionary.json"))

    if not os.path.isfile(sentences_csv):
        raise FileNotFoundError(
            f"sentences.csv not found at {sentences_csv}\n"
            "Run Stage 1 first."
        )

    top_words = run_fit(Path(sentences_csv), cache_dir)
    run_finalize(top_words, output_path)


def run_stage_3() -> None:
    _banner("Stage 3 — Sentiment assignment")
    from absa.sentiment_assignment import run
    run()


def run_stage_4() -> None:
    _banner("Stage 4 — Rating impact model + SHAP")
    from rating_impact.model import run as run_model
    run_model()

    _banner("Stage 4 — Model evaluation")
    from rating_impact.evaluate import run as run_eval
    run_eval()


def run_stage_5() -> None:
    _banner("Stage 5 — ChromaDB ingestion")
    from agent.ingest import run
    run()


# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hotel Review Intelligence — full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_pipeline.py            # run all stages\n"
            "  python scripts/run_pipeline.py --from 3   # resume from stage 3\n"
            "  python scripts/run_pipeline.py --only 5   # re-run ingest only\n"
        ),
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="from_stage", type=int, metavar="N",
        choices=range(1, 6),
        help="Resume pipeline from stage N (1–5). Skips earlier stages.",
    )
    group.add_argument(
        "--only", dest="only_stage", type=int, metavar="N",
        choices=range(1, 6),
        help="Run only stage N (1–5).",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

STAGE_RUNNERS = {
    1: run_stage_1,
    2: run_stage_2,
    3: run_stage_3,
    4: run_stage_4,
    5: run_stage_5,
}

STAGE_NAMES = {
    1: "Text preprocessing",
    2: "Aspect extraction (LDA)",
    3: "Sentiment assignment",
    4: "Rating impact model + evaluation",
    5: "ChromaDB ingestion",
}


def main() -> None:
    args = parse_args()

    if args.only_stage:
        stages_to_run = [args.only_stage]
    elif args.from_stage:
        stages_to_run = list(range(args.from_stage, 6))
    else:
        stages_to_run = list(range(1, 6))

    total_start = time.time()
    print(f"\nPipeline: running stages {stages_to_run}")

    for stage in stages_to_run:
        stage_start = time.time()
        try:
            STAGE_RUNNERS[stage]()
            print(f"\n[Stage {stage}] Done in {_elapsed(stage_start)}")
        except Exception as e:
            print(f"\n[Stage {stage}] FAILED after {_elapsed(stage_start)}: {e}")
            raise

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {_elapsed(total_start)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
