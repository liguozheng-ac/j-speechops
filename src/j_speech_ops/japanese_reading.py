"""Model-neutral reading provider and OpenJTalk Stage 6 baseline."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReadingInfrastructureError(RuntimeError):
    """The reading backend or its dictionary cannot initialize."""


class ReadingSampleError(ValueError):
    """The initialized backend could not produce one usable reading."""


class ReadingOverrideEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    surface: str = Field(min_length=1)
    reading_kana: str = Field(min_length=1)

    @model_validator(mode="after")
    def values_must_not_be_blank(self) -> "ReadingOverrideEntry":
        if not self.surface.strip() or not self.reading_kana.strip():
            raise ValueError("override surface and reading_kana must not be blank")
        if re.search(r"[ァ-ヺ]", self.reading_kana) is None:
            raise ValueError("override reading_kana must contain Katakana")
        if re.search(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９]", self.reading_kana):
            raise ValueError("override reading_kana must not contain Latin or digits")
        return self


class ReadingOverrideConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    overrides: tuple[ReadingOverrideEntry, ...] = ()

    @model_validator(mode="after")
    def surfaces_must_be_unique(self) -> "ReadingOverrideConfig":
        surfaces = [item.surface for item in self.overrides]
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("override surfaces must be unique")
        return self


class AppliedReadingOverride(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    surface: str
    reading_kana: str
    source: str
    occurrences: int = Field(gt=0)


class ReadingProviderInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    provider_version: str
    package_name: str
    package_version: str
    dictionary_identity: str | None = None
    dictionary_version: str | None = None
    settings: dict[str, bool | str] = Field(default_factory=dict)


class ReadingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    provider: str
    provider_version: str
    input_text: str
    baseline_reading_kana: str
    reading_kana: str
    override_count: int = Field(ge=0)
    applied_overrides: tuple[AppliedReadingOverride, ...] = ()
    override_fingerprint_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class ReadingOverrides:
    def __init__(
        self,
        config: ReadingOverrideConfig,
        *,
        fingerprint_sha256: str,
        path: Path,
    ) -> None:
        self.config = config
        self.fingerprint_sha256 = fingerprint_sha256
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "ReadingOverrides":
        override_path = Path(path)
        try:
            content = override_path.read_bytes()
            payload = json.loads(content.decode("utf-8"))
            config = ReadingOverrideConfig.model_validate(payload)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ReadingInfrastructureError(
                f"invalid reading override config {override_path}: {exc}"
            ) from exc
        return cls(
            config,
            fingerprint_sha256=hashlib.sha256(content).hexdigest(),
            path=override_path,
        )

    @classmethod
    def empty(cls) -> "ReadingOverrides":
        return cls(
            ReadingOverrideConfig(version="none", source="none"),
            fingerprint_sha256=hashlib.sha256(b"").hexdigest(),
            path=Path("<none>"),
        )

    def apply(self, text: str) -> tuple[str, tuple[AppliedReadingOverride, ...]]:
        if not self.config.overrides:
            return text, ()
        entries = {
            item.surface: item.reading_kana for item in self.config.overrides
        }
        surfaces = sorted(entries, key=lambda value: (-len(value), value))
        pattern = re.compile("|".join(re.escape(value) for value in surfaces))
        counts = {surface: 0 for surface in surfaces}

        def replace(match: re.Match[str]) -> str:
            surface = match.group(0)
            counts[surface] += 1
            return entries[surface]

        replaced = pattern.sub(replace, text)
        source = f"{self.config.source} ({self.path.as_posix()})"
        applied = tuple(
            AppliedReadingOverride(
                surface=surface,
                reading_kana=entries[surface],
                source=source,
                occurrences=counts[surface],
            )
            for surface in surfaces
            if counts[surface]
        )
        return replaced, applied


class ReadingProvider(Protocol):
    @property
    def info(self) -> ReadingProviderInfo: ...

    def prepare(
        self, sample_id: str, text: str, overrides: ReadingOverrides
    ) -> ReadingResult: ...


class OpenJTalkReadingProvider:
    """OpenJTalk frontend only; no HTSEngine, phoneme export, or pitch model."""

    _SETTINGS: dict[str, bool | str] = {
        "kana": True,
        "run_marine": False,
        "use_tsqyomi": False,
        "use_sudachi_kanji_yomi": False,
        "predict_nani": False,
        "normalize_mode": "None",
    }

    def __init__(self) -> None:
        try:
            self._backend = importlib.import_module("pyopenjtalk")
            package_version = importlib.metadata.version("pyopenjtalk-plus")
            dictionary_path = self._backend.OPEN_JTALK_DICT_DIR
            if isinstance(dictionary_path, bytes):
                dictionary_path = dictionary_path.decode("utf-8")
            dictionary = Path(dictionary_path)
            if not dictionary.is_dir() or not (dictionary / "sys.dic").is_file():
                raise FileNotFoundError("bundled OpenJTalk dictionary is unavailable")
            probe = self._g2p("動作確認")
            self._validate_reading(probe)
        except Exception as exc:
            raise ReadingInfrastructureError(
                f"OpenJTalk frontend initialization failed: {exc}"
            ) from exc
        self._info = ReadingProviderInfo(
            provider="openjtalk-frontend",
            provider_version=package_version,
            package_name="pyopenjtalk-plus",
            package_version=package_version,
            dictionary_identity="pyopenjtalk-plus bundled customized OpenJTalk/NAIST-jdic",
            dictionary_version=None,
            settings=dict(self._SETTINGS),
        )

    @property
    def info(self) -> ReadingProviderInfo:
        return self._info

    def _g2p(self, text: str) -> str:
        result = self._backend.g2p(text, **self._SETTINGS)
        if not isinstance(result, str):
            raise ReadingSampleError("OpenJTalk returned a non-string reading")
        return result

    @staticmethod
    def _validate_reading(reading: str) -> None:
        if not reading.strip() or re.search(r"[ァ-ヺ]", reading) is None:
            raise ReadingSampleError("OpenJTalk did not produce a Katakana reading")
        if re.search(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９]", reading):
            raise ReadingSampleError(
                "OpenJTalk reading contains unresolved Latin letters or digits"
            )

    def prepare(
        self, sample_id: str, text: str, overrides: ReadingOverrides
    ) -> ReadingResult:
        if not text.strip():
            raise ReadingSampleError("reading input is empty or whitespace-only")
        try:
            baseline = self._g2p(text)
            replaced, applied = overrides.apply(text)
            final = self._g2p(replaced) if applied else baseline
            self._validate_reading(final)
        except ReadingSampleError:
            raise
        except (RuntimeError, TypeError, UnicodeError, ValueError) as exc:
            raise ReadingSampleError(f"OpenJTalk frontend failed: {exc}") from exc
        return ReadingResult(
            sample_id=sample_id,
            provider=self.info.provider,
            provider_version=self.info.provider_version,
            input_text=text,
            baseline_reading_kana=baseline,
            reading_kana=final,
            override_count=sum(item.occurrences for item in applied),
            applied_overrides=applied,
            override_fingerprint_sha256=overrides.fingerprint_sha256,
        )
