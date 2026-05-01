# app.py — Module D (UI): HF Spaces entry point; launches the Gradio UI with the demo hotel subset.
#
# Input:  outputs/hotel_names.json, chromadb_demo/ or demo_vectors/ (numpy fallback)
# Output: None (serves Gradio UI)

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the UI module before overriding any globals
import ui.app as _ui_app

# Filter hotel dropdown to the demo subset (hotels actually in this Space's ChromaDB).
# hotel_names.json in the Space repo is the filtered list from build_demo_db.py.
_hotel_names_path = os.path.join(_ROOT, "outputs", "hotel_names.json")
if os.path.isfile(_hotel_names_path):
    with open(_hotel_names_path, encoding="utf-8") as _f:
        _demo_hotels = json.load(_f)
    _ui_app.HOTEL_OPTIONS = ["All Hotels"] + sorted(_demo_hotels)

from ui.app import build_ui

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
