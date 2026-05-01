# app.py — Module D (UI): Gradio chatbot with hotel dropdown, SHAP impact chart, multi-turn chat, and cited sources panel.
#
# Input:  outputs/hotel_names.json, chromadb/ (via agent graph)
# Output: None (serves Gradio UI at localhost:7860)

from __future__ import annotations

import json
import os
import sys
import uuid

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.graph import run_query
from paths import SHAP_SUMMARY


# ── Hotel / SHAP data ─────────────────────────────────────────────────────────

def _load_shap() -> dict:
    if not os.path.isfile(SHAP_SUMMARY):
        return {}
    with open(SHAP_SUMMARY, encoding="utf-8") as f:
        return {e["hotel_name"]: e for e in json.load(f)}

SHAP_DATA     = _load_shap()
HOTEL_OPTIONS = ["All Hotels"] + sorted(h for h in SHAP_DATA if h != "__global__")


# ── SHAP bar chart ────────────────────────────────────────────────────────────

def make_shap_chart(display_name: str) -> plt.Figure:
    key    = "__global__" if display_name == "All Hotels" else display_name
    entry  = SHAP_DATA.get(key, SHAP_DATA.get("__global__", {}))
    impacts: dict[str, float] = entry.get("aspect_impacts", {})

    if not impacts:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No SHAP data available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Sort by absolute impact descending
    sorted_items = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    labels  = [a.capitalize() for a, _ in sorted_items]
    values  = [v for _, v in sorted_items]
    colors  = ["#4CAF50" if v >= 0 else "#E53935" for v in values]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.55)
    ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean SHAP value  (positive = raises score)", fontsize=8)
    ax.set_title(f"Rating drivers — {display_name}", fontsize=9, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ── Citation formatter ────────────────────────────────────────────────────────

def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return "*No sources retrieved for this response.*"
    lines = ["**Retrieved sources**\n"]
    for i, c in enumerate(citations, 1):
        asp  = c.get("aspect", "")
        sent = c.get("sentiment", "")
        seg  = c.get("reviewer_segment", "")
        scr  = c.get("reviewer_score", 0)
        sim  = c.get("similarity_score", 0)
        txt  = c.get("text", "")
        lines.append(
            f"**[{i}]** `{asp}` · `{sent}` · `{seg}` "
            f"· reviewer score {scr} · similarity {sim:.2f}  \n"
            f"> {txt}\n"
        )
    return "\n".join(lines)


# ── Chat handler ──────────────────────────────────────────────────────────────

def chat(
    message: str,
    history: list,
    hotel_display: str,
    thread_id: str,
):
    hotel_key = "__global__" if hotel_display == "All Hotels" else hotel_display

    result = run_query(
        query=message,
        hotel_name=hotel_key,
        thread_id=thread_id,
    )

    response   = result.get("response", "")
    citations  = result.get("citations", [])
    unresolved = result.get("hotel_unresolved", False)
    insuf      = result.get("insufficient_data", False)

    # Inline warnings
    warnings = []
    if unresolved:
        warnings.append(
            "⚠️ Hotel name matched with low confidence — results may be from a nearby match."
        )
    if insuf:
        warnings.append(
            "ℹ️ Limited review data for this hotel — SHAP rankings may be less reliable."
        )

    full_response = (
        "\n".join(warnings) + "\n\n" + response if warnings else response
    )

    new_history = history + [{"role": "user",      "content": message},
                             {"role": "assistant",  "content": full_response}]
    return new_history, new_history, _format_citations(citations), ""


def reset_session(hotel_display: str):
    """Called when the hotel dropdown changes — clears chat and rotates thread_id."""
    return [], [], "*No sources yet.*", str(uuid.uuid4()), make_shap_chart(hotel_display)


# ── UI assembly ───────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Hotel Review Intelligence") as demo:

        # Per-session state
        thread_id_state = gr.State(value=str(uuid.uuid4()))
        history_state   = gr.State(value=[])

        # Header
        gr.Markdown(
            "# Hotel Review Intelligence\n"
            "Ask about guest sentiment, what aspects drive ratings, "
            "and where improvements matter most."
        )

        # Hotel selector
        hotel_dropdown = gr.Dropdown(
            choices=HOTEL_OPTIONS,
            value="All Hotels",
            label="Select hotel (or ask about all hotels)",
        )

        with gr.Row():
            # Left: SHAP chart
            with gr.Column(scale=1, min_width=300):
                shap_plot = gr.Plot(
                    value=make_shap_chart("All Hotels"),
                    label="Rating drivers (SHAP)",
                )
                gr.Markdown(
                    "<small>Bar chart shows mean SHAP values from the XGBoost "
                    "rating model. Green = raises ratings, red = lowers them.</small>"
                )

            # Right: chatbot
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=420,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder=(
                            "e.g. What do guests complain about most? "
                            "Which aspect should this hotel fix first?"
                        ),
                        label="",
                        scale=5,
                        autofocus=True,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

        # Citations panel
        citations_md = gr.Markdown("*No sources yet.*")

        # ── Wiring ─────────────────────────────────────────────────────────────

        chat_inputs  = [msg_input, history_state, hotel_dropdown, thread_id_state]
        chat_outputs = [chatbot, history_state, citations_md, msg_input]

        send_btn.click(fn=chat, inputs=chat_inputs, outputs=chat_outputs)
        msg_input.submit(fn=chat, inputs=chat_inputs, outputs=chat_outputs)

        hotel_dropdown.change(
            fn=reset_session,
            inputs=[hotel_dropdown],
            outputs=[chatbot, history_state, citations_md, thread_id_state, shap_plot],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
