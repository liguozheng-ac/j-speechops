# Stage 7.5 — Project Relocation Handoff

## Canonical local project root

```text
D:\AI-Projects\J-SpeechOps
```

All Stage 8 and later development must use this directory. The former Codex workspace remains available only as a retained backup and external model-cache location:

```text
<temporary-codex-workspace>
```

The old workspace was not deleted.

## Relocation boundary

The project was copied rather than moved. Source, tests, documentation, configuration, examples, schemas, data documentation, generated smoke outputs, small work fixtures, and `.git` metadata were copied. The old Windows `.venv` was excluded and rebuilt from `pyproject.toml` with Python 3.14.7.

The 2.88 GiB Hugging Face-format `Systran/faster-whisper-large-v3` cache was intentionally not duplicated. GPU validation from the canonical root reused this retained cache with `local_files_only=1`:

```text
<temporary-codex-workspace>\work\stage3_models
```

No model was moved, deleted, or downloaded.

## Verification baseline

- Editable package: `j-speech-ops 0.7.0`
- Python: `3.14.7`
- Full tests: `138 passed, 1 skipped`
- Opt-in large-v3 CUDA integration: `1 passed`
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- Stage 6 frontend amount smoke: verified
- Stage 7 routing smoke: PASS=1, REVIEW=2, DROP=1, failed=0
- SpeechSample schema SHA-256: `DEC2C98402E4433DF64E0B0F2280A73C7EE8CF3EE03232D7BCC8EC2EF7B70435`
- TTSTextSample schema SHA-256: `169E87DFA910A64BEF2E1200EB94927C96857065C76629D7FB07CB107DE74EDE`

## Git condition carried forward

The copied `.git` directory resolves to the canonical root. The repository is on `master`, has no commits, has no configured remote, and all project source files remain untracked. No repository was initialized, no commit was created, and nothing was pushed during relocation.
