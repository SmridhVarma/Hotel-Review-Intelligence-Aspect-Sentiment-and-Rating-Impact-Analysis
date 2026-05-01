# aspect_extraction.py — Stage 2 | Module A (ABSA): Trains a 6-topic LDA model on the sentence corpus to produce keyword vocabularies for each hotel aspect.
#
# Input:  outputs/sentences.csv
# Output: outputs/aspect_dictionary.json

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aspect_extraction")


# --------------------------------------------------------------------------- #
# Config — defaults
# --------------------------------------------------------------------------- #
N_TOPICS = 6
TOP_N_WORDS = 20
RANDOM_STATE = 42

# CountVectorizer parameters tuned for ~836k short sentences.
#   min_df=20: drop terms appearing in fewer than 20 sentences. With 836k docs
#       this filters typos, rare proper nouns, and one-off slang while keeping
#       any term used more than ~once per 40k sentences.
#   max_df=0.5: drop terms appearing in more than 50% of sentences. Hotel-review
#       sentences share a lot of generic verbs ("was", "were", "had") that even
#       after stopword removal can dominate; this caps their influence.
#   ngram_range=(1, 2): unigrams + bigrams. Bigrams like "front desk",
#       "air conditioning", "noise level" carry strong aspect signal that
#       single tokens miss.
#   token_pattern: alphabetic tokens of length >=3 (digits and 1-2 char tokens
#       are noise for topic modeling).
VECTORIZER_PARAMS = dict(
    min_df=20,
    max_df=0.5,
    ngram_range=(1, 2),
    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    max_features=50_000,
)

# LDA parameters.
#   learning_method='online': mini-batch variational Bayes. Required for a
#       corpus this size — 'batch' would load the full DTM into memory each
#       iteration and be ~5-10x slower.
#   batch_size=4096: balances throughput and convergence noise.
#   max_iter=15: online LDA converges fast on short, topically-narrow text;
#       diminishing returns past ~15 passes for this corpus.
#   doc_topic_prior / topic_word_prior left at sklearn default (1/n_topics).
LDA_PARAMS = dict(
    n_components=N_TOPICS,
    learning_method="online",
    batch_size=4096,
    max_iter=15,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Domain stopwords — generic hotel-stay filler that appears across all
# aspects and washes out topic separation. Critically, we keep words that
# ARE aspect names (room, staff, food, location, noise) and aspect-adjacent
# nouns (bed, bath, breakfast) so LDA can anchor topics to them.
DOMAIN_STOPWORDS = {
    "hotel", "stay", "stayed", "staying", "night", "nights", "day", "days",
    "time", "times", "place", "places", "thing", "things", "way", "lot",
    "bit", "kind", "sort", "review", "reviews", "guest", "guests",
    "would", "could", "should", "will", "shall", "may", "might", "must",
    "really", "very", "much", "well", "good", "bad", "nice", "great", "ok",
    "okay", "fine", "yes", "no", "not", "got", "get", "got", "go", "going",
    "went", "come", "came", "say", "said", "told", "tell", "see", "seen",
    "look", "looked", "made", "make", "take", "took", "give", "gave", "use",
    "used", "definitely", "absolutely", "totally", "actually", "basically",
    "literally", "probably", "maybe", "perhaps", "though", "however",
    "although", "even", "still", "always", "never", "ever", "also", "just",
    "one", "two", "three", "first", "second", "next", "last", "every", "each",
    "everything", "anything", "something", "nothing",
    # pronouns / aux verbs that survive lemmatization
    "us", "we", "they", "them", "their", "there", "here", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "him", "her", "his",
    "hers", "my", "mine", "your", "yours", "our", "ours",
}


# --------------------------------------------------------------------------- #
# TOPIC -> ASPECT MAPPING (operator-curated, fill in after Phase A)
# --------------------------------------------------------------------------- #
#
# After running phase A (`--phase fit`), inspect the top-N keyword printouts,
# decide which LDA topic index corresponds to each aspect, and fill in the
# integers below. Then run `--phase finalize` (or rerun `--phase all`).
#
# Set every value to a valid topic index in [0, N_TOPICS - 1]. A topic may be
# mapped to at most one aspect; if two topics overlap, pick the cleaner one
# and review the discarded topic's words for inclusion in OVERFLOW_KEYWORDS.
#
# Example after Phase A:
#     TOPIC_TO_ASPECT = {
#         "cleanliness": 3,
#         "staff":       1,
#         "location":    0,
#         "noise":       4,
#         "food":        2,
#         "room":        5,
#     }
#
TOPIC_TO_ASPECT: Dict[str, Optional[int]] = {
    "cleanliness": 0,
    "staff":       2,
    "location":    1,
    "noise":       4,
    "food":        3,
    "room":        5,
}

# Hand-curated keyword fallbacks. Merged into the LDA vocabulary at finalize
# time so the dictionary is robust even if a topic is noisy. These are
# semantic seeds, not the final word list.
ASPECT_SEED_KEYWORDS: Dict[str, List[str]] = {
    "cleanliness": [
        "clean", "cleanliness", "dirty", "stain", "stained", "dust", "dusty",
        "hygiene", "hygienic", "filthy", "spotless", "tidy", "smell", "smelly",
        "musty", "mold", "mould", "hair", "hairs", "sheets", "towels",
    ],
    "staff": [
        "staff", "service", "reception", "receptionist", "concierge", "manager",
        "friendly", "helpful", "rude", "polite", "professional", "attentive",
        "welcoming", "front desk", "housekeeping", "team", "employee",
    ],
    "location": [
        "location", "located", "central", "centre", "center", "near", "close",
        "walking distance", "metro", "station", "transport", "airport",
        "city centre", "neighborhood", "neighbourhood", "area", "view",
    ],
    "noise": [
        "noise", "noisy", "quiet", "loud", "soundproof", "soundproofing",
        "thin walls", "street noise", "traffic", "music", "construction",
        "sleep", "sleeping", "disturbed", "peaceful",
    ],
    "food": [
        "food", "breakfast", "buffet", "restaurant", "menu", "meal", "dinner",
        "lunch", "coffee", "tea", "drink", "drinks", "bar", "wine", "tasty",
        "delicious", "fresh", "cuisine", "dining",
    ],
    "room": [
        "room", "bed", "bedroom", "pillow", "mattress", "bathroom", "shower",
        "bath", "toilet", "tv", "wifi", "ac", "air conditioning", "heating",
        "balcony", "spacious", "small", "tiny", "comfortable",
    ],
}


# --------------------------------------------------------------------------- #
# Text preprocessing
# --------------------------------------------------------------------------- #
class TextCleaner:
    """spaCy-backed cleaner: lowercase, depunctuate, lemmatize, drop stopwords.

    spaCy is preferred over NLTK here because:
      - Its lemmatizer is rule-based + lookup, far more accurate on
        verbs/adjectives than NLTK's WordNet-only lemmatizer.
      - `nlp.pipe(..., n_process=...)` processes 836k short docs in
        ~5-10 minutes on a laptop.

    Components disabled (parser, NER, attribute_ruler, etc.) since we only
    need the tokenizer + lemmatizer + tagger.
    """

    def __init__(self, model: str = "en_core_web_sm",
                 extra_stopwords: Optional[set] = None):
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spaCy is required. Install with:\n"
                "    pip install spacy\n"
                "    python -m spacy download en_core_web_sm"
            ) from e

        try:
            self.nlp = spacy.load(model, disable=["parser", "ner",
                                                  "attribute_ruler"])
        except OSError as e:
            raise OSError(
                f"spaCy model '{model}' not found. Install with:\n"
                f"    python -m spacy download {model}"
            ) from e

        # Combine spaCy's stopwords with our domain list.
        self.stopwords = set(self.nlp.Defaults.stop_words)
        if extra_stopwords:
            self.stopwords.update(extra_stopwords)

        self._punct_re = re.compile(f"[{re.escape(string.punctuation)}]")
        self._ws_re = re.compile(r"\s+")

    def _basic_clean(self, text: str) -> str:
        """Lowercase, strip punctuation/digits, collapse whitespace."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = self._punct_re.sub(" ", text)
        text = re.sub(r"\d+", " ", text)
        text = self._ws_re.sub(" ", text).strip()
        return text

    def clean_corpus(self, texts: List[str], batch_size: int = 1000,
                     n_process: int = 1) -> List[str]:
        """Vectorized cleaning over a list of sentences.

        Returns a list of space-joined lemma strings, ready for CountVectorizer.
        Empty / unparseable inputs return ''.
        """
        # Phase 1: regex-level normalisation (fast, no spaCy needed).
        pre = [self._basic_clean(t) for t in texts]

        # Phase 2: spaCy lemmatization in batches.
        cleaned: List[str] = []
        log.info("Lemmatizing %d documents (batch_size=%d, n_process=%d)...",
                 len(pre), batch_size, n_process)

        for i, doc in enumerate(self.nlp.pipe(pre, batch_size=batch_size,
                                              n_process=n_process)):
            tokens = []
            for tok in doc:
                if tok.is_space or tok.is_punct:
                    continue
                lemma = tok.lemma_.strip().lower()
                if not lemma or len(lemma) < 3:
                    continue
                if lemma in self.stopwords:
                    continue
                if not lemma.isalpha():
                    continue
                tokens.append(lemma)
            cleaned.append(" ".join(tokens))

            if (i + 1) % 50_000 == 0:
                log.info("  lemmatized %d / %d", i + 1, len(pre))

        return cleaned


# --------------------------------------------------------------------------- #
# LDA fitting
# --------------------------------------------------------------------------- #
def build_dtm(texts: List[str]) -> Tuple[CountVectorizer, "scipy.sparse.csr_matrix"]:
    """Fit a CountVectorizer and return (vectorizer, document-term matrix)."""
    log.info("Building Document-Term Matrix with params: %s", VECTORIZER_PARAMS)
    vectorizer = CountVectorizer(**VECTORIZER_PARAMS)
    dtm = vectorizer.fit_transform(texts)
    log.info("DTM shape: %s | vocabulary size: %d",
             dtm.shape, len(vectorizer.get_feature_names_out()))
    return vectorizer, dtm


def fit_lda(dtm) -> LatentDirichletAllocation:
    """Fit LDA with the configured parameters."""
    log.info("Fitting LDA: %s", LDA_PARAMS)
    lda = LatentDirichletAllocation(**LDA_PARAMS)
    lda.fit(dtm)
    log.info("LDA fit complete. Final perplexity: %.2f", lda.perplexity(dtm))
    return lda


def get_top_words(lda: LatentDirichletAllocation, vectorizer: CountVectorizer,
                  top_n: int = TOP_N_WORDS) -> Dict[int, List[str]]:
    """Return {topic_idx: [top_n words]} sorted by topic-word probability."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_words: Dict[int, List[str]] = {}
    for topic_idx, topic in enumerate(lda.components_):
        # argsort ascending; take last top_n and reverse for descending order.
        top_indices = topic.argsort()[-top_n:][::-1]
        top_words[topic_idx] = feature_names[top_indices].tolist()
    return top_words


def print_top_words(top_words: Dict[int, List[str]]) -> None:
    """Pretty-print the top-N words per topic for human inspection."""
    print("\n" + "=" * 78)
    print("LDA TOPIC TOP WORDS")
    print("=" * 78)
    print("Inspect each topic and decide which one matches each aspect.")
    print("Then fill in TOPIC_TO_ASPECT at the top of this script and rerun")
    print("with --phase finalize (or --phase all).\n")
    for topic_idx, words in top_words.items():
        print(f"Topic {topic_idx}:")
        # Two columns of 10 for readability.
        col1, col2 = words[:10], words[10:20]
        for left, right in zip(col1, col2):
            print(f"    {left:<22}    {right}")
        print()
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------- #
# Aspect dictionary assembly
# --------------------------------------------------------------------------- #
def validate_mapping(mapping: Dict[str, Optional[int]]) -> None:
    """Sanity-check the operator-supplied TOPIC_TO_ASPECT dict."""
    missing = [a for a, t in mapping.items() if t is None]
    if missing:
        raise ValueError(
            f"TOPIC_TO_ASPECT is incomplete. Fill in topics for: {missing}\n"
            f"After running Phase A, edit TOPIC_TO_ASPECT at the top of "
            f"src/absa/aspect_extraction.py with the topic indices you choose."
        )

    # Check ranges.
    for aspect, topic_idx in mapping.items():
        if not (0 <= topic_idx < N_TOPICS):
            raise ValueError(
                f"Aspect '{aspect}' mapped to invalid topic {topic_idx}. "
                f"Must be in [0, {N_TOPICS - 1}]."
            )

    # Check uniqueness.
    used = list(mapping.values())
    if len(set(used)) != len(used):
        log.warning(
            "Two or more aspects share the same topic index: %s. This is "
            "allowed but unusual — confirm before continuing.", used
        )


def build_aspect_dictionary(
    top_words: Dict[int, List[str]],
    mapping: Dict[str, int],
    seeds: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Merge LDA topic words with hand-curated seeds, dedupe, preserve order.

    LDA words come first (they reflect actual corpus usage), seeds top up
    coverage for terms LDA may have ranked below the top-N cutoff.
    """
    aspect_dict: Dict[str, List[str]] = {}
    for aspect, topic_idx in mapping.items():
        lda_words = top_words.get(topic_idx, [])
        seed_words = seeds.get(aspect, [])
        seen = set()
        merged = []
        for w in list(lda_words) + list(seed_words):
            wl = w.lower().strip()
            if wl and wl not in seen:
                seen.add(wl)
                merged.append(wl)
        aspect_dict[aspect] = merged
    return aspect_dict


def save_artifact(aspect_dict: Dict[str, List[str]],
                  top_words: Dict[int, List[str]],
                  mapping: Dict[str, int],
                  output_path: Path) -> None:
    """Write the aspect dictionary plus provenance metadata to JSON."""
    payload = {
        "aspects": aspect_dict,
        "metadata": {
            "n_topics": N_TOPICS,
            "top_n_words": TOP_N_WORDS,
            "topic_to_aspect": mapping,
            "raw_topic_top_words": {str(k): v for k, v in top_words.items()},
            "vectorizer_params": {k: (v if not callable(v) else str(v))
                                  for k, v in VECTORIZER_PARAMS.items()},
            "lda_params": {k: v for k, v in LDA_PARAMS.items()},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("Wrote aspect dictionary -> %s", output_path)


# --------------------------------------------------------------------------- #
# Pipeline phases
# --------------------------------------------------------------------------- #
def run_fit(input_csv: Path, cache_dir: Path,
            n_process: int = 1) -> Dict[int, List[str]]:
    """Phase A: load data, clean, fit LDA, dump top words. Cache intermediates."""
    log.info("Loading sentences from %s", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    if "sentence" not in df.columns:
        raise ValueError(
            f"Expected column 'sentence' in {input_csv}. "
            f"Found: {list(df.columns)}"
        )
    sentences = df["sentence"].fillna("").astype(str).tolist()
    log.info("Loaded %d sentences", len(sentences))

    # Clean.
    cleaner = TextCleaner(extra_stopwords=DOMAIN_STOPWORDS)
    cleaned = cleaner.clean_corpus(sentences, batch_size=1000,
                                   n_process=n_process)

    # Drop docs that became empty after cleaning (otherwise CountVectorizer
    # warns and they contribute zero rows to topic estimates anyway).
    nonempty_mask = [bool(t.strip()) for t in cleaned]
    cleaned_nonempty = [t for t, m in zip(cleaned, nonempty_mask) if m]
    log.info("Non-empty cleaned docs: %d / %d (%.1f%%)",
             len(cleaned_nonempty), len(cleaned),
             100 * len(cleaned_nonempty) / max(1, len(cleaned)))

    # DTM + LDA.
    vectorizer, dtm = build_dtm(cleaned_nonempty)
    lda = fit_lda(dtm)
    top_words = get_top_words(lda, vectorizer, TOP_N_WORDS)

    # Cache top words to disk so finalize can run later without refitting.
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "lda_top_words.json"
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in top_words.items()}, f, indent=2)
    log.info("Cached LDA top words -> %s", cache_path)

    print_top_words(top_words)
    return top_words


def run_finalize(top_words: Dict[int, List[str]], output_path: Path) -> None:
    """Phase B: validate the operator mapping and emit the JSON artifact."""
    validate_mapping(TOPIC_TO_ASPECT)
    mapping_int: Dict[str, int] = {k: int(v) for k, v in TOPIC_TO_ASPECT.items()}
    aspect_dict = build_aspect_dictionary(top_words, mapping_int,
                                          ASPECT_SEED_KEYWORDS)

    log.info("Final aspect dictionary preview:")
    for aspect, words in aspect_dict.items():
        log.info("  %-12s (%2d words): %s", aspect, len(words),
                 ", ".join(words[:10]) + (" ..." if len(words) > 10 else ""))

    save_artifact(aspect_dict, top_words, mapping_int, output_path)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 — LDA aspect vocabulary extraction."
    )
    p.add_argument(
        "--input", type=Path, default=Path("outputs/sentences.csv"),
        help="Path to Stage 1 sentences.csv (default: outputs/sentences.csv)"
    )
    p.add_argument(
        "--output", type=Path, default=Path("outputs/aspect_dictionary.json"),
        help="Where to write the aspect dictionary JSON."
    )
    p.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/_lda_cache"),
        help="Directory for cached LDA top-words between phases."
    )
    p.add_argument(
        "--phase", choices=["fit", "finalize", "all"], default="all",
        help=("'fit' = train LDA & dump top words for review; "
              "'finalize' = apply TOPIC_TO_ASPECT mapping & write JSON; "
              "'all' = run both (requires TOPIC_TO_ASPECT to be filled in).")
    )
    p.add_argument(
        "--n-process", type=int, default=1,
        help="spaCy n_process for nlp.pipe(). Set >1 on multi-core machines."
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.phase in ("fit", "all"):
        if not args.input.exists():
            log.error("Input file not found: %s", args.input)
            return 1
        top_words = run_fit(args.input, args.cache_dir,
                            n_process=args.n_process)
    else:
        # Finalize-only: load cached top words.
        cache_path = args.cache_dir / "lda_top_words.json"
        if not cache_path.exists():
            log.error(
                "No cached top words at %s. Run --phase fit first.", cache_path
            )
            return 1
        with cache_path.open("r", encoding="utf-8") as f:
            top_words = {int(k): v for k, v in json.load(f).items()}

    if args.phase in ("finalize", "all"):
        # If running --phase all and the operator hasn't filled in the mapping
        # yet, exit gracefully after Phase A rather than raising.
        if any(v is None for v in TOPIC_TO_ASPECT.values()):
            log.warning(
                "TOPIC_TO_ASPECT is not yet filled in. "
                "Phase A complete; cached top words are at %s. "
                "Edit TOPIC_TO_ASPECT in this file, then run --phase finalize.",
                args.cache_dir / "lda_top_words.json"
            )
            return 0
        run_finalize(top_words, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
