# mermaid_diagram_generator.py — scripts utility: Generates Mermaid diagram source for the pipeline and agent DAG.
#
# Input:  None
# Output: Prints Mermaid diagram markdown to stdout, or updates docs/ARCHITECTURE.md in-place with --update flag

from __future__ import annotations

import argparse
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Diagram definitions
# ---------------------------------------------------------------------------

PIPELINE_DIAGRAM = """\
```mermaid
flowchart TD
    DATA(["data/data.xlsx<br>515k reviews, 1492 hotels"])

    DATA --> S1["Stage 1 — preprocess.py<br>Sentence splitting + segment tagging"]
    S1 --> SENT(["sentences.csv<br>~836k sentences"])
    S1 --> CLEAN([clean_reviews_stage1.csv])

    SENT --> S2["Stage 2 — aspect_extraction.py<br>LDA topic model → aspect keywords"]
    S2 --> ADICT([aspect_dictionary.json])

    SENT --> S3["Stage 3 — sentiment_assignment.py<br>Keyword matching + sentiment assignment"]
    ADICT --> S3
    S3 --> ASENT(["aspect_sentences.csv<br>~735k labelled sentences"])
    S3 --> RFEAT([review_features.csv])

    RFEAT --> S4["Stage 4 — model.py + evaluate.py<br>Linear Regression + XGBoost + SHAP"]
    S4 --> SHAP([shap_summary.json])
    S4 --> EVALS([evaluation_report.json])

    ASENT --> S5["Stage 5 — ingest.py<br>text-embedding-3-small → ChromaDB"]
    SHAP --> S5
    S5 --> CHROMA[("chromadb/<br>evidence_store · summary_store")]

    CHROMA --> RUNTIME["Stage 5 runtime — graph.py<br>LangGraph DAG · 8 nodes"]
    RUNTIME --> UI["src/ui/app.py<br>Gradio chatbot"]
```"""

AGENT_DAG_DIAGRAM = """\
```mermaid
flowchart TD
    IN([User query + hotel_name]) --> QC["query_classifier<br>GPT-4o · classifies type, direction, aspects"]
    QC --> SF["segment_filter<br>Reviewer segment detection"]
    SF --> COND{query_type?}
    COND -- "prioritization / mismatch" --> SR["summary_retriever<br>summary_store metadata lookup"]
    COND -- "evidence / segment" --> HY["hyde_expander<br>GPT-4o · generate hypothetical review"]
    HY --> ER["evidence_retriever<br>evidence_store similarity search"]
    SR --> CM["context_merger<br>Merge chunks · flag low confidence"]
    ER --> CM
    CM --> RG["response_generator<br>GPT-4o · final answer + citations"]
    RG --> SM["state_manager<br>Append turn · clear cache on topic shift"]
    SM --> OUT([Response + citations])
```"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_markdown(section: str) -> str:
    parts = []
    if section in ("pipeline", "all"):
        parts.append("## Pipeline diagram\n\n" + PIPELINE_DIAGRAM)
    if section in ("agent", "all"):
        parts.append("## Agent DAG\n\n" + AGENT_DAG_DIAGRAM)
    return "\n\n".join(parts)


def update_architecture(section: str) -> None:
    """Re-embed diagrams between their marker comments in docs/ARCHITECTURE.md."""
    arch_path = os.path.join(_ROOT, "docs", "ARCHITECTURE.md")
    with open(arch_path, encoding="utf-8") as f:
        content = f.read()

    markers = {
        "pipeline": ("<!-- PIPELINE_DIAGRAM_START -->", "<!-- PIPELINE_DIAGRAM_END -->"),
        "agent":    ("<!-- AGENT_DAG_START -->",         "<!-- AGENT_DAG_END -->"),
    }
    diagrams = {
        "pipeline": PIPELINE_DIAGRAM,
        "agent":    AGENT_DAG_DIAGRAM,
    }

    targets = ["pipeline", "agent"] if section == "all" else [section]
    changed = False
    for target in targets:
        start_marker, end_marker = markers[target]
        new_block = f"{start_marker}\n{diagrams[target]}\n{end_marker}"
        pattern = re.escape(start_marker) + r".*?" + re.escape(end_marker)
        if re.search(pattern, content, flags=re.DOTALL):
            content = re.sub(pattern, new_block, content, flags=re.DOTALL)
            changed = True
        else:
            print(
                f"Warning: markers for '{target}' not found in ARCHITECTURE.md — nothing updated.",
                file=sys.stderr,
            )

    if changed:
        with open(arch_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {arch_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Mermaid diagram source for the pipeline and agent DAG."
    )
    parser.add_argument(
        "--section",
        choices=["pipeline", "agent", "all"],
        default="all",
        help="Which diagram to generate (default: all)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update docs/ARCHITECTURE.md in-place instead of printing to stdout",
    )
    args = parser.parse_args()

    if args.update:
        update_architecture(args.section)
    else:
        print(get_markdown(args.section))


if __name__ == "__main__":
    main()
