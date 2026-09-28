# J-SpeechOps Architecture Overview

## 1. Project Overview

J-SpeechOps is a quality-control and release workflow system for Japanese speech-data production.

It connects text preparation, speech generation, artifact validation, machine-generated quality evidence, human review, and release management without treating any single model output as authoritative. Each stage produces explicit records that can be inspected, reproduced, and traced to its inputs and policy.

The repository also contains an independent speech-data preparation path for validating, segmenting, transcribing, and curating source recordings. The speech-data and TTS-release paths share engineering principles and contracts, but neither is an implicit prerequisite for the other.

## 2. Pipeline Architecture

```text
Text Input
    |
    v
Text Preparation and Curation
    |
    v
TTS Generation
    |
    v
Audio Post-processing
    |
    v
Artifact Creation
    |
    v
Audio QA
    |
    v
Pronunciation Evidence and Regression
    |
    v
Human Review
    |
    v
Release
```

The boundaries are deliberate:

- generation creates an artifact but does not declare quality;
- audio QA checks integrity and content consistency but does not grant human approval;
- pronunciation analysis records evidence and regression signals but does not claim linguistic truth;
- human review resolves subjective or uncertain language questions;
- the release gate verifies that the reviewed artifact and every referenced evidence object are still current.

## 3. Core Design Principles

### Evidence-driven Quality Control

Quality decisions are based on inspectable artifact evidence, QA results, structured issue records, and policy provenance. Machine disagreement becomes review evidence rather than an invented confidence score.

### Human-in-the-loop

Machines detect objective failures, identify risks, compare observations, and assemble review context. Human reviewers make the final language and listening judgment. Human approval cannot repair a missing, corrupt, silent, hash-invalid, or provenance-inconsistent artifact.

### Provenance-first

Artifacts retain generation identity, effective configuration, processing policy, content hash, upstream evidence references, and review history. Regeneration or policy changes create new identities, preventing approval from silently carrying over to different audio.

### Conservative State Transitions

`PASS`, `REVIEW`, `APPROVE`, and `RELEASED` represent different decisions. A machine `REVIEW` is resolvable by a qualified reviewer; it is not equivalent to `DROP`. Release requires a fresh approval bound to the current artifact and evidence context.

## 4. Current Limitations

The current system does not provide:

- pitch-accent validation;
- MOS evaluation;
- automatic naturalness scoring;
- production deployment or serving infrastructure;
- reviewer authentication or authorization;
- a graphical review interface;
- proof that machine transcription or reading tools are phonetic ground truth.

These limitations are explicit so downstream users do not infer guarantees beyond the recorded evidence.
