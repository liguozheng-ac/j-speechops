"""Deterministic Japanese character-form normalization for Stage 6."""

from __future__ import annotations

import importlib.metadata
import re
from typing import Literal

import jaconv
from pydantic import BaseModel, ConfigDict, Field

JAPANESE_TEXT_PREP_POLICY_VERSION = "japanese-text-prep-v1.1"

_THOUSANDS_GROUPED_INTEGER = re.compile(
    r"(?<![0-9,.])[0-9]{1,3}(?:,[0-9]{3})+(?![0-9,.])"
)


class JapaneseNormalizationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(
        default=JAPANESE_TEXT_PREP_POLICY_VERSION, min_length=1
    )
    unicode_mode: Literal["NFKC"] = "NFKC"
    collapse_whitespace: bool = True
    normalize_latin_hyphen: bool = True
    remove_numeric_thousands_separators: bool = True


class NormalizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str
    raw_text: str
    normalized_text: str
    changed: bool
    transformations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    jaconv_version: str


class NormalizationSampleError(ValueError):
    """A single text could not be normalized under the active policy."""


class JapaneseNormalizer:
    """Character-level normalization; it does not generate a reading."""

    def __init__(self, policy: JapaneseNormalizationPolicy | None = None) -> None:
        self.policy = policy or JapaneseNormalizationPolicy()
        self.jaconv_version = importlib.metadata.version("jaconv")

    def normalize(self, raw_text: str) -> NormalizationResult:
        if not raw_text.strip():
            raise NormalizationSampleError("raw text is empty or whitespace-only")

        transformations: list[str] = []
        normalized = jaconv.normalize(raw_text, self.policy.unicode_mode)
        if normalized != raw_text:
            transformations.append("jaconv_unicode_width_normalization")

        if self.policy.remove_numeric_thousands_separators:
            numeric_grouping = _THOUSANDS_GROUPED_INTEGER.sub(
                lambda match: match.group(0).replace(",", ""), normalized
            )
            if numeric_grouping != normalized:
                transformations.append("numeric_thousands_separator_removed")
            normalized = numeric_grouping

        if self.policy.normalize_latin_hyphen:
            latin_hyphen = re.sub(
                r"(?<=[A-Za-z0-9])ー(?=[A-Za-z0-9])", "-", normalized
            )
            if latin_hyphen != normalized:
                transformations.append("latin_hyphen_normalization")
            normalized = latin_hyphen

        if self.policy.collapse_whitespace:
            whitespace = re.sub(r"\s+", " ", normalized).strip()
            if whitespace != normalized:
                transformations.append("whitespace_normalization")
            normalized = whitespace

        if not normalized:
            raise NormalizationSampleError("normalization produced empty text")
        return NormalizationResult(
            policy_version=self.policy.policy_version,
            raw_text=raw_text,
            normalized_text=normalized,
            changed=normalized != raw_text,
            transformations=tuple(transformations),
            jaconv_version=self.jaconv_version,
        )
