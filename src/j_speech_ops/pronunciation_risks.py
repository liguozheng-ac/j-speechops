"""Versioned pronunciation-risk watchlist for Stage 7 routing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .japanese_reading import AppliedReadingOverride


class PronunciationRiskEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    surface: str = Field(min_length=1)
    category: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    note: str | None = None

    @model_validator(mode="after")
    def values_must_not_be_blank(self) -> "PronunciationRiskEntry":
        for name in ("surface", "category", "reason"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        if self.note is not None and not self.note.strip():
            raise ValueError("note must be omitted or nonblank")
        return self


class PronunciationRiskWatchlistConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    entries: tuple[PronunciationRiskEntry, ...] = ()

    @model_validator(mode="after")
    def entries_must_be_valid(self) -> "PronunciationRiskWatchlistConfig":
        if not self.version.strip() or not self.source.strip():
            raise ValueError("watchlist version and source must not be blank")
        surfaces = [entry.surface for entry in self.entries]
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("pronunciation-risk surfaces must be unique")
        return self


class PronunciationRiskMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    surface: str
    category: str
    reason: str
    note: str | None = None
    occurrence_count: int = Field(gt=0)


class PronunciationRiskWatchlist:
    def __init__(
        self,
        config: PronunciationRiskWatchlistConfig,
        *,
        fingerprint_sha256: str,
        path: Path,
    ) -> None:
        self.config = config
        self.fingerprint_sha256 = fingerprint_sha256
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "PronunciationRiskWatchlist":
        watchlist_path = Path(path)
        try:
            content = watchlist_path.read_bytes()
            payload = json.loads(content.decode("utf-8"))
            config = PronunciationRiskWatchlistConfig.model_validate(payload)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(
                f"invalid pronunciation risk watchlist {watchlist_path}: {exc}"
            ) from exc
        return cls(
            config,
            fingerprint_sha256=hashlib.sha256(content).hexdigest(),
            path=watchlist_path,
        )

    @classmethod
    def empty(cls) -> "PronunciationRiskWatchlist":
        return cls(
            PronunciationRiskWatchlistConfig(
                version="none", source="no pronunciation risk watchlist"
            ),
            fingerprint_sha256=hashlib.sha256(b"").hexdigest(),
            path=Path("<none>"),
        )

    def match(
        self,
        *,
        raw_text: str,
        normalized_text: str | None,
        applied_overrides: tuple[AppliedReadingOverride, ...],
    ) -> tuple[tuple[PronunciationRiskMatch, ...], tuple[PronunciationRiskMatch, ...]]:
        """Return unresolved and override-resolved matches in config order."""
        override_occurrences = {
            item.surface: item.occurrences for item in applied_overrides
        }
        unresolved: list[PronunciationRiskMatch] = []
        resolved: list[PronunciationRiskMatch] = []
        normalized = normalized_text or ""
        for entry in self.config.entries:
            occurrence_count = max(
                raw_text.count(entry.surface), normalized.count(entry.surface)
            )
            if occurrence_count == 0:
                continue
            match = PronunciationRiskMatch(
                surface=entry.surface,
                category=entry.category,
                reason=entry.reason,
                note=entry.note,
                occurrence_count=occurrence_count,
            )
            if override_occurrences.get(entry.surface, 0) >= occurrence_count:
                resolved.append(match)
            else:
                unresolved.append(match)
        return tuple(unresolved), tuple(resolved)
