# Stage 10B — Human Review & Release Gate

## Purpose

Stage 10B adds the explicit human authority and release boundary after Stage 9 content QA and Stage 10A pronunciation/regression evidence.

```text
Stage 10A evidence and regression
  -> mandatory HumanReviewQueueItem
  -> HumanReviewRecord (APPROVE / REWORK / REJECT)
  -> independent ReleaseGateResult
  -> ReleasedAudioArtifact
```

Machine QA PASS, absence of Stage 10A regression evidence, human approval, and release are four different states. Under the current policy every release candidate requires a real human final review. There is no automatic approval or release path.

## Contracts

- `HumanReviewQueueItem`: immutable review context, hard blockers, structured machine-review issues, and references to the Stage 8 artifact, Stage 9 result, and Stage 10A evidence/regression result.
- `MachineReviewIssue`: stable issue identity, source, code, evidence reference, and summary for a human-resolvable Stage 9/10A/regression REVIEW condition.
- `HumanReviewRecord`: append-only decision provenance bound to one exact review-context fingerprint, including structured `resolved_issue_ids`.
- `HumanReviewBatchSummary`: deterministic queue counts.
- `ReleaseGateResult`: independent policy evaluation with explicit denial reasons and review freshness.
- `ReleasedAudioArtifact`: released-manifest entry linking source text, synthesis, artifact hash, audio post-processing provenance, QA, pronunciation evidence, regression, human review, and gate identities.
- `ReleaseBatchSummary`: deterministic released/not-released and missing/stale-review counts.

Reviewer identifiers are provenance strings only. J-SpeechOps does not authenticate them.

## Human review semantics

Every queue item includes `MANDATORY_FINAL_REVIEW`, even when Stage 9 is PASS and Stage 10A reports no machine-routed review condition.

- `APPROVE`: permits release-gate evaluation but does not itself release the artifact.
- `REWORK`: denies release and may identify an upstream return target; Stage 10B never modifies upstream data automatically.
- `REJECT`: denies the current artifact entry.

Negative decisions require at least one structured reason: `CONTENT`, `PRONUNCIATION`, `PROSODY`, `NATURALNESS`, `AUDIO_ARTIFACT`, `VOICE_STYLE`, or `OTHER`. Notes remain optional. There are no numeric quality, pronunciation, naturalness, confidence, MOS, or pitch-accent scores.

Machine REVIEW is not DROP. Stage 9 content mismatch, Stage 10A missing/unresolved pronunciation evidence, and detected pronunciation regression become `MachineReviewIssue` records. A current human `APPROVE` may resolve them only by explicitly listing every issue ID in `resolved_issue_ids`; notes alone do not resolve an issue. Partial acknowledgement leaves the omitted issue visible in the gate result and denies release.

Objective artifact failures are separate hard blockers. Missing, corrupt, silent, hash-invalid, invalid-WAV, failed-QA, or DROP evidence cannot be overridden by ordinary human APPROVE and requires upstream rework.

## Stale-review protection

The queue computes a deterministic review-context fingerprint over the current artifact/run, audio SHA-256, audio post-processing provenance, source/normalized/synthesis text, expected/risk references, confirmed-override fingerprint, Stage 9 identity/policy, Stage 10A evidence/regression identities and policies, synthesis configuration, and human-review policy.

A review stores that fingerprint and all critical identities. Release evaluation selects the latest review for the logical case. Any mismatch makes it `STALE`, including a changed artifact, run, text, override provenance, Stage 9 result, Stage 10A evidence, regression result, or review policy. A stale approval is never inherited.

## Release policy

`RELEASED` requires all of the following:

- no hard artifact/Stage 9 blocker exists;
- the latest human review is current and `APPROVE`;
- every human-resolvable machine REVIEW issue is explicitly present in that review's `resolved_issue_ids`;
- queue, review, gate, and active release-policy identities agree.

Stage 9 REVIEW, Stage 10A REVIEW, and regression REVIEW are therefore human-routing states rather than permanent denials. REWORK, REJECT, a missing/stale review, an unresolved issue, or any hard blocker yields `NOT_RELEASED` with structured denial reasons. There is no `--force-release` option. Release creates a new `ReleasedAudioArtifact`; it does not mutate `GeneratedAudioArtifact`.

## CLI usage

Prepare the review queue:

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.human_review_release_cli prepare-review `
    --artifact-manifest outputs\stage8_tts_generation\integration_smoke\generated_audio_manifest.jsonl `
    --synthesis-runs outputs\stage8_tts_generation\integration_smoke\synthesis_runs.jsonl `
    --qa-results outputs\stage9_generated_audio_qa\integration_smoke\generated_audio_qa_results.jsonl `
    --pronunciation-evidence outputs\stage10a_pronunciation_regression\integration_smoke\pronunciation_evidence.jsonl `
    --regression-results outputs\stage10a_pronunciation_regression\integration_smoke\pronunciation_regression_results.jsonl `
    --output-dir outputs\stage10b_human_review_release\integration_smoke
```

After a real person listens to the referenced WAV, record exactly that person's decision:

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.human_review_release_cli record-review `
    --queue outputs\stage10b_human_review_release\integration_smoke\human_review_queue.jsonl `
    --records outputs\stage10b_human_review_release\integration_smoke\human_review_records.jsonl `
    --artifact-id <artifact-run-id> `
    --reviewer-id <reviewer-identifier> `
    --decision approve
```

When the queue item contains machine review issues, add every issue that the reviewer actually resolved:

```powershell
... record-review --decision approve `
    --resolve-issue <issue-id-1> `
    --resolve-issue <issue-id-2>
```

An APPROVE without all required `--resolve-issue` values remains `NOT_RELEASED`.

For a negative decision, supply one or more `--reason` values and optional notes/return target:

```powershell
... record-review --decision rework --reason pronunciation `
    --rework-target pronunciation_override --notes "Describe the audible issue"
```

Evaluate the gate, then build the release manifest:

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.human_review_release_cli evaluate-release `
    --queue outputs\stage10b_human_review_release\integration_smoke\human_review_queue.jsonl `
    --records outputs\stage10b_human_review_release\integration_smoke\human_review_records.jsonl `
    --output-dir outputs\stage10b_human_review_release\integration_smoke

.\.venv\Scripts\python.exe -m j_speech_ops.human_review_release_cli build-release-manifest `
    --queue outputs\stage10b_human_review_release\integration_smoke\human_review_queue.jsonl `
    --records outputs\stage10b_human_review_release\integration_smoke\human_review_records.jsonl `
    --gate-results outputs\stage10b_human_review_release\integration_smoke\release_gate_results.jsonl `
    --output outputs\stage10b_human_review_release\integration_smoke\released_audio_manifest.jsonl
```

Always rerun `evaluate-release` after adding or changing a review. The manifest builder refuses stale gate/review provenance.

## Outputs

- `human_review_queue.jsonl`
- `human_review_records.jsonl`
- `human_review_summary.json`
- `release_gate_results.jsonl`
- `released_audio_manifest.jsonl`
- `release_summary.json`

Queue, gate, and summary outputs are deterministic for identical inputs. Human review and released records retain real timestamps and therefore are not expected to be byte-identical across distinct real actions.

## Real human workflow

1. Generate the queue without modifying Stage 8–10A outputs.
2. A real reviewer listens to each exact `audio_path` and checks the supplied evidence and machine-review issue IDs.
3. Record APPROVE, REWORK, or REJECT under the reviewer's explicit identifier; an APPROVE explicitly lists every machine issue it resolves.
4. Rerun the release gate.
5. Build the release manifest only from the new gate results.

Automated agents, test fixtures, Stage 9 PASS, and Stage 10A no-regression results are not human reviewers.

## Limitations

- No automatic pitch-accent validation.
- No automatic naturalness or prosody judgment.
- No reviewer authentication, accounts, or RBAC.
- No Web UI.
- No production deployment or distribution workflow.
- No cryptographic signature or external audit-log service.
- Release policy operates on existing Stage 9/10A evidence and does not replace listening.
