# aspect_extraction.py — Stage 2: Aspect Extraction via LDA
#
# Purpose:
#   Applies LDA topic modeling to the sentence corpus to build a keyword
#   dictionary for 6 target hotel service aspects.
#
# Input:
#   outputs/sentences.csv
#     sentence (str): sentence text to model
#
# Output:
#   outputs/aspect_dictionary.json
#     Schema: { aspect_name: [keyword, ...] }
#     Keys:   cleanliness | staff | location | noise | food | room
#     Example: { "cleanliness": ["clean", "dirty", "mold", "smell"], ... }

import os
import sys
import json

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paths import SENTENCES, ASPECT_DICTIONARY


def load_sentences() -> pd.DataFrame:
    """Load the Stage 1 sentence corpus."""
    return pd.read_csv(SENTENCES, low_memory=False)


# 836,795 rows — one sentence per row, split from positive/negative review fields.
# Columns: review_id, hotel_name, reviewer_score, reviewer_segment,
#          review_date, sentence, source_field, source_polarity
df = load_sentences()
