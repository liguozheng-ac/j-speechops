# Local repository placement

## Project root

Run development and verification from the checked-out project root. A Python virtual environment is local state: create it from `pyproject.toml` after cloning or copying the repository instead of transferring an existing Windows environment between directories.

## Repository boundary

The repository contains source, tests, documentation, configuration, examples, schemas, and small deterministic fixtures. Runtime outputs, work files, model weights, large audio files, caches, and virtual environments remain outside version control under the rules in `.gitignore`.

Whisper and TTS checkpoints are external runtime dependencies. Supply their locations through the documented command-line or environment settings; reusable source code must not depend on a machine-specific absolute path.

## Relocation checklist

When relocating a checkout:

1. Copy the repository, including `.git`, before retiring the source checkout.
2. Rebuild the environment from `pyproject.toml`.
3. Keep model storage external and configure its path explicitly.
4. Run `pip check`, the full test suite, and the opt-in hardware smoke tests that apply to the machine.
5. Confirm ignored local resources remain untracked.

No model download or deletion is required merely because the repository location changes.

## Stable contracts

- SpeechSample schema SHA-256: `DEC2C98402E4433DF64E0B0F2280A73C7EE8CF3EE03232D7BCC8EC2EF7B70435`
- TTSTextSample schema SHA-256: `169E87DFA910A64BEF2E1200EB94927C96857065C76629D7FB07CB107DE74EDE`
