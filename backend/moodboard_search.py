from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable


_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_FIELD_WEIGHTS = {
    "style_axes": 7.0,
    "keywords": 6.5,
    "title": 4.5,
    "conditioning_notes": 3.5,
    "prompt_guidance": 3.0,
    "taste_profile": 1.5,
    "suggest_style_axes": 7.0,
    "suggest_conditioning_notes": 3.5,
    "suggest_prompt_guidance": 3.0,
}
_SUGGESTION_FIELDS = frozenset(
    {
        "suggest_style_axes",
        "suggest_conditioning_notes",
        "suggest_prompt_guidance",
    }
)
_CATALOG_FIELDS = frozenset(
    {
        "title",
        "taste_profile",
        "keywords",
        "style_axes",
        "conditioning_notes",
        "prompt_guidance",
    }
)
# Extend suggestion vocabulary here. Keep tokens grouped by visual-style role;
# arbitrary subject nouns must never be added merely to improve recall.
STYLE_TOKEN_GROUPS: dict[str, frozenset[str]] = {
    "medium_and_movements": frozenset(
        {
            "academic", "acrylic", "airbrush", "art", "artisanal", "bauhaus",
            "baroque", "brutalism", "charcoal", "cinematic", "collage",
            "constructivism", "cubism", "dadaism", "documentary", "editorial",
            "engraving", "etching", "expressionism", "filmic", "futurism",
            "gestural", "gouache", "graphite", "illustration", "impressionism",
            "ink", "linocut", "lithograph", "minimalism", "modernism",
            "naturalism", "noir", "oil", "paint", "painterly", "pastel",
            "photocopy", "photographic", "photography", "pictorialist", "pop",
            "postmodernism", "print", "realism", "renaissance", "risograph",
            "rococo", "romanticism", "screenprint", "sketch", "surreal",
            "surrealism", "surrealist", "vaporwave", "watercolor", "woodcut",
            "zine",
        }
    ),
    "lighting": frozenset(
        {
            "ambient", "backlight", "backlit", "binary", "chiaroscuro",
            "diffuse", "diffusion", "directional", "dramatic", "flash", "glow",
            "glowing", "hard-light", "highlight", "highlights", "light",
            "lighting", "low-key", "luminance", "neon", "practical", "raking",
            "rim", "shadow", "shadows", "shadowplay", "softbox", "sunbeam",
            "sunbeams", "volumetric",
        }
    ),
    "palette_and_color": frozenset(
        {
            "acid", "amber", "black", "blue", "bronze", "chromatic", "color",
            "colorful", "colour", "cold", "cool", "crimson", "cyan", "desaturated",
            "duotone", "golden", "gradient", "green", "harmony", "honey",
            "indigo", "monochrome", "monochromatic", "muted", "obsidian",
            "orange", "palette", "pink", "red", "saturated", "sepia", "slate",
            "stark", "teal", "temperature", "tonality", "tone", "tonal", "tones",
            "vibrant", "warm", "warmth", "yellow",
        }
    ),
    "texture_and_material": frozenset(
        {
            "blur", "chatter", "contour", "decay", "distressed", "dots",
            "faint", "finish", "fog", "grain", "graininess", "grit", "halftone", "haze",
            "imperfection", "marks", "material", "noise", "reflective",
            "reflectivity", "refraction", "sculptural", "sheen", "smear",
            "smears", "surface", "tactile", "texture", "veiled",
        }
    ),
    "composition_and_framing": frozenset(
        {
            "angle", "asymmetrical", "balance", "centered", "center-weighted",
            "brightness", "circularity", "clarity", "clean", "composition", "depth",
            "dissolve", "dynamic", "edge", "field", "flattened", "focus",
            "focused", "form", "forms", "fractured", "framing", "geometry", "geometric",
            "graphic", "iconic", "lines", "linearity", "minimal", "minimalist",
            "medium", "movement", "negative", "pattern", "perspective", "portrait", "precision",
            "scale", "shapes", "silhouette", "space", "spatial", "staging",
            "symmetrical",
        }
    ),
    "camera_lens_and_film": frozenset(
        {
            "analog", "aperture", "bokeh", "camera", "candor",
            "cinematographic", "exposure", "film", "fisheye", "focal", "lens",
            "long-exposure", "motion", "snapshot", "wide-angle",
        }
    ),
    "era_and_design": frozenset(
        {
            "alpine", "antique", "classic", "contemporary", "deco", "design", "era",
            "mid-century", "modern", "nostalgic", "poster", "punk", "retro",
            "retro-futurist", "retrofuturist", "vintage", "whimsical",
        }
    ),
    "mood_and_atmosphere": frozenset(
        {
            "atmosphere", "atmospheric", "desolate", "dreamlike", "dreamy", "emotional",
            "energy", "existential", "ethereal", "hush", "intimate", "intimacy",
            "liminal", "longing", "mood", "moody", "quiet", "soft", "stillness",
            "subversive", "tension", "theatrical", "vast",
        }
    ),
    "rendering_techniques": frozenset(
        {
            "contrast", "contrasting", "definition", "digital", "distorted",
            "effect", "effects", "glitch", "hyper-reflective", "mirrored",
            "pixel-sorted", "refracted", "render", "rendering", "shading",
            "sharp", "starburst", "starbursts", "technique", "warp",
        }
    ),
    "named_styles": frozenset(
        {
            "afrofuturism", "cyberpunk", "hyperrealism", "maximalism",
            "neo-expressionism", "photorealism", "retrofuturism", "solarpunk",
            "steampunk", "synthwave", "ukiyo-e",
        }
    ),
    "style_modifiers": frozenset(
        {
            "bold", "dark", "heavy", "high", "low", "style",
        }
    ),
}
STYLE_TOKENS = frozenset().union(*STYLE_TOKEN_GROUPS.values())
# Multi-token or ambiguous named styles live here. Matching accepts spaces or
# hyphens and indexes the canonical hyphenated phrase as one style token.
STYLE_PHRASES = frozenset(
    {
        "art deco",
        "art nouveau",
        "film grain",
        "film noir",
        "heavy film grain",
        "high contrast",
        "low poly",
        "muted palette",
        "neo expressionism",
        "pixel art",
        "soft focus",
        "ukiyo e",
    }
)
STYLE_PHRASE_TOKENS = frozenset(
    phrase.replace(" ", "-") for phrase in STYLE_PHRASES
)
MODIFIER_STYLE_TOKENS = frozenset(
    {
        "art", "atmosphere", "atmospheric", "balance", "bold", "clean",
        "color", "colour", "composition", "contrast", "contrasting", "dark",
        "depth", "dramatic", "field", "focus", "focused", "form", "forms",
        "framing", "heavy", "high", "light", "lighting", "low", "medium",
        "minimal", "mood", "moody", "movement", "muted", "negative", "pattern",
        "render", "rendering", "scale", "shapes", "soft", "space", "spatial",
        "staging", "style", "technique", "tone", "tonal", "tones", "warm",
    }
)
ANCHOR_STYLE_TOKENS = STYLE_TOKENS - MODIFIER_STYLE_TOKENS
_STYLE_PHRASE_PATTERNS = tuple(
    (
        phrase.replace(" ", "-"),
        phrase.split()[0],
        re.compile(
            r"(?<![a-z0-9])"
            + r"[-\s]+".join(re.escape(part) for part in phrase.split())
            + r"(?![a-z0-9])"
        ),
    )
    for phrase in STYLE_PHRASES
)
_STYLE_MORPH_ROOTS = {
    "brutal", "constructiv", "cottage", "cub", "cyber", "dada", "dream",
    "expression", "futur", "goblin", "impression", "minimal", "modern",
    "natural", "norm", "painter", "postmodern", "real", "romantic", "surreal",
    "weird",
}
_STYLE_CONNECTORS = frozenset(
    {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
        "of", "on", "or", "the", "to", "with", "against", "across", "between",
        "over", "through", "under", "via", "while", "without",
    }
)
_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def _ascii_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[\u2010\u2011]", "-", text)
    return re.sub(r"[\u2012-\u2015\u2212]", " ", text)


def normalize_tokens(value: object) -> list[str]:
    """Return boundary-safe words plus compounds from explicit hyphen phrases."""
    result: list[str] = []
    for match in _WORD_RE.finditer(_ascii_text(value)):
        compound = match.group(0).strip("-")
        parts = [part for part in compound.split("-") if part]
        result.extend(parts)
        if len(parts) > 1:
            result.append("-".join(parts))
    return result


def format_matched_reason(
    cues: Iterable[object],
    *,
    max_cues: int = 3,
    max_cue_chars: int = 48,
    max_reason_chars: int = 180,
) -> str:
    """Format short, deterministic UI evidence without leaking guidance prose."""
    rendered: list[str] = []
    seen: set[str] = set()
    for raw_cue in cues:
        cue = re.sub(r"\s+", " ", str(raw_cue or "")).strip(" \t\r\n.,;:·-|")
        if not cue:
            continue
        if len(cue) > max_cue_chars:
            shortened = cue[: max_cue_chars + 1].rsplit(" ", 1)[0].strip()
            cue = (shortened or cue[:max_cue_chars]).rstrip(" \t\r\n.,;:·-|")
        key = cue.casefold()
        if not cue or key in seen:
            continue
        seen.add(key)
        rendered.append(cue)
        if len(rendered) >= max_cues:
            break
    reason = f"Matched: {' · '.join(rendered)}" if rendered else "Matched style cues"
    return reason[:max_reason_chars].rstrip()


def _word_sequence(value: object) -> tuple[str, ...]:
    return tuple(token for token in normalize_tokens(value) if "-" not in token)


@lru_cache(maxsize=8192)
def _stem(token: str) -> str:
    if "-" in token:
        return "-".join(_stem(part) for part in token.split("-"))
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("es") and not token.endswith(("ses", "xes")):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if left == right:
        return True
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    index_short = index_long = differences = 0
    while index_short < len(shorter) and index_long < len(longer):
        if shorter[index_short] == longer[index_long]:
            index_short += 1
            index_long += 1
            continue
        differences += 1
        index_long += 1
        if differences > 1:
            return False
    return True


def _values(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


@lru_cache(maxsize=8192)
def _is_style_token(token: str) -> bool:
    normalized = str(token or "").casefold().strip("-")
    if not normalized:
        return False
    if (
        normalized in STYLE_TOKENS
        or normalized in STYLE_PHRASE_TOKENS
        or _stem(normalized) in STYLE_TOKENS
    ):
        return True
    if "-" in normalized:
        return all(_is_style_token(part) for part in normalized.split("-"))
    suffix_lengths = {"ism": 3, "esque": 5, "core": 4}
    for suffix, suffix_length in suffix_lengths.items():
        if not normalized.endswith(suffix):
            continue
        root = normalized[:-suffix_length]
        minimum = 3 if suffix == "ism" else 4
        return len(root) >= minimum and root in _STYLE_MORPH_ROOTS
    return False


@lru_cache(maxsize=8192)
def _is_anchor_style_token(token: str) -> bool:
    normalized = str(token or "").casefold().strip("-")
    if normalized in STYLE_PHRASE_TOKENS:
        return True
    if normalized in ANCHOR_STYLE_TOKENS:
        return True
    if "-" in normalized:
        return any(_is_anchor_style_token(part) for part in normalized.split("-"))
    suffix_lengths = {"ism": 3, "esque": 5, "core": 4}
    return any(
        normalized.endswith(suffix)
        and normalized[:-suffix_length] in _STYLE_MORPH_ROOTS
        for suffix, suffix_length in suffix_lengths.items()
    )


def _curated_phrase_tokens_from_text(text: str) -> tuple[str, ...]:
    matches: list[tuple[int, str]] = []
    for phrase_token, first_word, pattern in _STYLE_PHRASE_PATTERNS:
        if first_word not in text:
            continue
        found = pattern.search(text)
        if found:
            matches.append((found.start(), phrase_token))
    return tuple(token for _position, token in sorted(matches))


def _curated_phrase_tokens(value: object) -> tuple[str, ...]:
    return _curated_phrase_tokens_from_text(_ascii_text(value))


def _analyze_fragment(
    value: object,
) -> tuple[
    set[str],
    tuple[str, ...],
    set[str],
    tuple[tuple[str, ...], ...],
]:
    text = _ascii_text(value)
    matches = tuple(_WORD_RE.finditer(text))
    raw_tokens: set[str] = set()
    raw_words: list[str] = []
    runs: list[tuple[str, ...]] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            runs.append(tuple(current))
            current.clear()

    for match in matches:
        token = match.group(0).strip("-")
        parts = tuple(part for part in token.split("-") if part)
        raw_tokens.update(parts)
        raw_words.extend(parts)
        if len(parts) > 1:
            raw_tokens.add("-".join(parts))
        if token in _STYLE_CONNECTORS:
            flush()
            continue
        if _is_style_token(token):
            current.append(token)
        else:
            flush()
    flush()

    tokens = {token for run in runs for token in run}
    tokens.update(
        f"{left}-{right}"
        for run in runs
        for left, right in zip(run, run[1:])
    )
    phrase_tokens = _curated_phrase_tokens_from_text(text)
    tokens.update(phrase_tokens)
    sequences = (*runs, *((token,) for token in phrase_tokens))
    return raw_tokens, tuple(raw_words), tokens, tuple(sequences)


def _analyze_style_text(
    value: object,
) -> tuple[set[str], tuple[tuple[str, ...], ...]]:
    _raw_tokens, _raw_words, style_tokens, style_sequences = _analyze_fragment(
        value
    )
    return style_tokens, style_sequences


def _safe_suggestion_guidance(item: dict, guidance: dict) -> dict[str, list[str]]:
    version = int(
        item.get("qwen_guidance_version")
        or guidance.get("guidance_version")
        or 0
    )
    if version < 2 and str(item.get("source") or "") != "andrometa":
        return {
            "style_axes": [],
            "conditioning_notes": [],
            "prompt_guidance": [],
        }
    axes = _values(guidance.get("style_axes"))
    notes = _values(guidance.get("conditioning_notes"))
    prompts = _values(guidance.get("prompt_guidance"))
    return {
        "style_axes": axes,
        "conditioning_notes": notes,
        "prompt_guidance": prompts,
    }


@dataclass(frozen=True)
class SearchResult:
    item_id: int
    score: float
    matched_cues: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Posting:
    item_id: int
    field: str


@dataclass
class _Document:
    item_id: int
    title: str
    style_tokens: set[str]


class MoodboardSearchIndex:
    """Immutable in-memory relevance index for parsed moodboard dictionaries."""

    def __init__(self, items: Iterable[dict]) -> None:
        self._documents: dict[int, _Document] = {}
        postings: dict[str, list[_Posting]] = defaultdict(list)
        bigram_postings: dict[tuple[str, str], list[_Posting]] = defaultdict(list)
        stem_postings: dict[str, set[str]] = defaultdict(set)
        document_frequency: dict[str, int] = defaultdict(int)

        for item in items:
            item_id = int(item["id"])
            guidance = item.get("qwen_guidance") or {}
            safe_guidance = _safe_suggestion_guidance(item, guidance)
            raw_fields = {
                "title": _values(item.get("title")),
                "taste_profile": _values(item.get("taste_profile")),
                "keywords": _values(item.get("keywords")),
                "style_axes": _values(guidance.get("style_axes")),
                "conditioning_notes": _values(guidance.get("conditioning_notes")),
                "prompt_guidance": _values(guidance.get("prompt_guidance")),
                "suggest_style_axes": safe_guidance["style_axes"],
                "suggest_conditioning_notes": safe_guidance["conditioning_notes"],
                "suggest_prompt_guidance": safe_guidance["prompt_guidance"],
            }
            all_tokens: set[str] = set()
            style_tokens: set[str] = set()
            fragment_cache: dict[
                str,
                tuple[
                    set[str],
                    tuple[str, ...],
                    set[str],
                    tuple[tuple[str, ...], ...],
                ],
            ] = {}

            def analyze(value: object):
                key = str(value or "")
                cached = fragment_cache.get(key)
                if cached is None:
                    cached = _analyze_fragment(key)
                    fragment_cache[key] = cached
                return cached

            for field, values in raw_fields.items():
                if field in _SUGGESTION_FIELDS:
                    analyses = tuple(analyze(value) for value in values)
                    tokens = {
                        token
                        for _raw_tokens, _raw_words, analyzed_tokens, _sequences in analyses
                        for token in analyzed_tokens
                    }
                    sequences = tuple(
                        sequence
                        for _raw_tokens, _raw_words, _analyzed_tokens, analyzed_sequences in analyses
                        for sequence in analyzed_sequences
                    )
                else:
                    analyses = tuple(analyze(value) for value in values)
                    sequences = tuple(
                        raw_words
                        for _raw_tokens, raw_words, _style_tokens, _style_sequences in analyses
                    )
                    tokens = {
                        token
                        for raw_tokens, _raw_words, _style_tokens, _style_sequences in analyses
                        for token in raw_tokens
                    }
                all_tokens.update(tokens)
                if field in _SUGGESTION_FIELDS:
                    style_tokens.update(tokens)
                for token in tokens:
                    postings[token].append(_Posting(item_id, field))
                    stem_postings[_stem(token)].add(token)
                for bigram in {
                    pair
                    for sequence in sequences
                    for pair in zip(sequence, sequence[1:])
                }:
                    bigram_postings[bigram].append(_Posting(item_id, field))
            for token in all_tokens:
                document_frequency[token] += 1
            self._documents[item_id] = _Document(
                item_id=item_id,
                title=str(item.get("title") or ""),
                style_tokens=style_tokens,
            )

        count = max(1, len(self._documents))
        self._postings = {token: tuple(values) for token, values in postings.items()}
        self._bigram_postings = {
            bigram: tuple(values) for bigram, values in bigram_postings.items()
        }
        self._stem_tokens = {stem: tuple(sorted(tokens)) for stem, tokens in stem_postings.items()}
        self._vocabulary = tuple(sorted(self._postings))
        self._idf = {
            token: math.log((count + 1) / (frequency + 0.5)) + 1.0
            for token, frequency in document_frequency.items()
        }

    def _matching_tokens(self, query_token: str) -> tuple[tuple[str, float], ...]:
        matches: dict[str, float] = {}
        if query_token in self._postings:
            matches[query_token] = 1.0
        for token in self._stem_tokens.get(_stem(query_token), ()):
            matches[token] = max(matches.get(token, 0.0), 0.92)
        if not matches and len(query_token) >= 5:
            for token in self._vocabulary:
                if token.startswith(query_token):
                    matches[token] = 0.78
        if not matches and len(query_token) >= 6:
            for token in self._vocabulary:
                if len(token) >= 6 and _edit_distance_at_most_one(query_token, token):
                    matches[token] = 0.62
        return tuple(matches.items())

    def _score(
        self,
        query: str,
        *,
        fields: frozenset[str] | None = None,
    ) -> tuple[
        dict[int, float],
        dict[int, set[str]],
        dict[int, set[str]],
        dict[int, set[str]],
    ]:
        query_tokens = [
            token
            for token in dict.fromkeys(normalize_tokens(query))
            if token not in _QUERY_STOPWORDS
        ]
        query_words = tuple(
            word for word in _word_sequence(query) if word not in _QUERY_STOPWORDS
        )
        analyzed_query_tokens, _query_style_sequences = _analyze_style_text(query)
        query_tokens.extend(
            token
            for token in analyzed_query_tokens
            if token not in query_tokens
        )
        scores: dict[int, float] = defaultdict(float)
        matched_query: dict[int, set[str]] = defaultdict(set)
        matched_index: dict[int, set[str]] = defaultdict(set)
        matched_anchors: dict[int, set[str]] = defaultdict(set)
        allowed = fields or _CATALOG_FIELDS

        for query_token in query_tokens:
            for indexed_token, match_factor in self._matching_tokens(query_token):
                for posting in self._postings.get(indexed_token, ()):
                    if posting.field not in allowed:
                        continue
                    scores[posting.item_id] += (
                        _FIELD_WEIGHTS[posting.field]
                        * self._idf[indexed_token]
                        * match_factor
                    )
                    matched_query[posting.item_id].add(query_token)
                    matched_index[posting.item_id].add(indexed_token)
                    if (
                        posting.field in _SUGGESTION_FIELDS
                        and _is_anchor_style_token(indexed_token)
                    ):
                        matched_anchors[posting.item_id].add(query_token)

        query_bigrams = set(zip(query_words, query_words[1:]))
        for bigram in query_bigrams:
            for posting in self._bigram_postings.get(bigram, ()):
                if posting.field in allowed and posting.item_id in scores:
                    scores[posting.item_id] += _FIELD_WEIGHTS[posting.field] * 1.35
        return scores, matched_query, matched_index, matched_anchors

    def search(self, query: str, *, style_only: bool = False) -> list[SearchResult]:
        if not str(query or "").strip():
            return []
        fields = _SUGGESTION_FIELDS if style_only else None
        scores, _, matched_index, _ = self._score(query, fields=fields)
        results = [
            SearchResult(item_id, round(score, 6), tuple(sorted(matched_index[item_id])))
            for item_id, score in scores.items()
            if score > 0
        ]
        results.sort(
            key=lambda result: (
                -result.score,
                self._documents[result.item_id].title.casefold(),
                result.item_id,
            )
        )
        return results

    def _query_cues(
        self,
        original_prompt: str,
        expanded_prompt: str,
        matched_tokens: set[str],
    ) -> tuple[str, ...]:
        cues: list[str] = []
        seen: set[str] = set()
        for prompt in (original_prompt, expanded_prompt):
            phrase_tokens = _curated_phrase_tokens(prompt)
            _tokens, sequences = _analyze_style_text(prompt)
            ordered_sequences = (
                *((token,) for token in phrase_tokens),
                *(sequence for sequence in sequences if sequence not in {(token,) for token in phrase_tokens}),
            )
            for sequence in ordered_sequences:
                matched = [token for token in sequence if token in matched_tokens]
                if not matched:
                    continue
                cue = " ".join(matched[:4]).replace("-", " ")
                if cue not in seen:
                    seen.add(cue)
                    cues.append(cue)
                if len(cues) == 3:
                    return tuple(cues)
        return tuple(cues)

    def suggest(
        self,
        original_prompt: str,
        expanded_prompt: str,
        *,
        favorite_ids: set[int] | None = None,
        limit: int = 12,
    ) -> list[SearchResult]:
        original_scores, original_matches, _, original_anchors = self._score(
            original_prompt, fields=_SUGGESTION_FIELDS
        )
        expanded_scores, expanded_matches, _, expanded_anchors = self._score(
            expanded_prompt, fields=_SUGGESTION_FIELDS
        )
        candidate_ids = set(original_anchors) | set(expanded_anchors)
        favorites = favorite_ids or set()
        combined: dict[int, float] = {}
        for item_id in candidate_ids:
            score = original_scores.get(item_id, 0.0) * 3.0 + expanded_scores.get(item_id, 0.0)
            if item_id in favorites:
                score = score * 1.08 + 0.05
            if score >= 4.0:
                combined[item_id] = score
        if not combined:
            return []

        selected: list[int] = []
        candidate_pool_size = max(max(0, limit) * 10, 120)
        ranked_candidates = sorted(
            combined,
            key=lambda item_id: (
                -combined[item_id],
                self._documents[item_id].title.casefold(),
                item_id,
            ),
        )[:candidate_pool_size]
        remaining = set(ranked_candidates)
        max_relevance = max(combined.values())
        while remaining and len(selected) < max(0, limit):
            def mmr_key(item_id: int) -> tuple[float, float, str, int]:
                relevance = combined[item_id]
                similarity = 0.0
                if selected:
                    tokens = self._documents[item_id].style_tokens
                    similarity = max(
                        len(tokens & self._documents[chosen].style_tokens)
                        / max(1, len(tokens | self._documents[chosen].style_tokens))
                        for chosen in selected
                    )
                mmr = 0.72 * relevance - 0.28 * similarity * max_relevance
                document = self._documents[item_id]
                return (
                    mmr,
                    relevance,
                    "".join(chr(0x10FFFF - ord(char)) for char in document.title.casefold()),
                    -item_id,
                )

            chosen = max(remaining, key=mmr_key)
            selected.append(chosen)
            remaining.remove(chosen)

        return [
            SearchResult(
                item_id,
                round(combined[item_id], 6),
                self._query_cues(
                    original_prompt,
                    expanded_prompt,
                    original_matches.get(item_id, set()) | expanded_matches.get(item_id, set()),
                ),
            )
            for item_id in selected
        ]


@dataclass(frozen=True)
class CachedMoodboardSearch:
    items: tuple[dict, ...]
    index: MoodboardSearchIndex
    generation: int


_cache_lock = threading.RLock()
_cache_condition = threading.Condition(_cache_lock)
_cache: dict[str, CachedMoodboardSearch] = {}
_generations: dict[str, int] = defaultdict(int)
_build_counts: dict[str, int] = defaultdict(int)
_building: dict[str, int] = {}


def _cache_key(db_path: Path) -> str:
    return str(Path(db_path).resolve())


def invalidate_moodboard_search_cache(db_path: Path) -> None:
    key = _cache_key(db_path)
    with _cache_condition:
        _generations[key] += 1
        _cache.pop(key, None)
        _cache_condition.notify_all()


def get_cached_moodboard_search(
    db_path: Path,
    loader: Callable[[], list[dict]],
) -> CachedMoodboardSearch:
    key = _cache_key(db_path)
    while True:
        with _cache_condition:
            generation = _generations[key]
            cached = _cache.get(key)
            if cached is not None and cached.generation == generation:
                return cached
            if _building.get(key) == generation:
                _cache_condition.wait()
                continue
            _building[key] = generation
        try:
            documents = tuple(loader())
            index = MoodboardSearchIndex(documents)
            items = tuple(
                {
                    "id": int(document["id"]),
                    "uuid": str(document.get("uuid") or ""),
                    "title": str(document.get("title") or ""),
                    "source": str(document.get("source") or "official"),
                }
                for document in documents
            )
            built = CachedMoodboardSearch(
                items, index, generation
            )
        except BaseException:
            with _cache_condition:
                if _building.get(key) == generation:
                    _building.pop(key, None)
                _cache_condition.notify_all()
            raise
        with _cache_condition:
            if _building.get(key) == generation:
                _building.pop(key, None)
            if _generations[key] != generation:
                _cache_condition.notify_all()
                continue
            existing = _cache.get(key)
            if existing is not None and existing.generation == generation:
                _cache_condition.notify_all()
                return existing
            _cache[key] = built
            _build_counts[key] += 1
            _cache_condition.notify_all()
            return built


def moodboard_search_cache_build_count(db_path: Path) -> int:
    with _cache_lock:
        return _build_counts[_cache_key(db_path)]
