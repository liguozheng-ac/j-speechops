# Synthetic release flow

These records illustrate the current JSON contracts and their ID links. Every name, hash, timestamp, transcript, and decision is fictional. `example.wav` does not exist: there is no audio asset, model execution, QA run, or human listening behind these files. The example is for reading the manifest structure, not for replaying the release CLI or claiming a real release.

| Step | Record | Link to the next step |
|---|---|---|
| Text input | [TTSTextSample](tts_text_sample.example.json) | `sample_id = sample_001` |
| Generation | [GeneratedAudioArtifact](generated_audio_artifact.example.json) | `run_id` and `audio_sha256` |
| Machine QA | [GeneratedAudioQAResult](generated_audio_qa.example.json) | `qa_id` and artifact hash |
| Pronunciation | [PronunciationEvidenceResult](pronunciation_evidence.example.json) and [PronunciationRegressionResult](pronunciation_regression.example.json) | Stage 9 QA ID and evidence ID |
| Review | [HumanReviewQueueItem](human_review_queue.example.json) and [HumanReviewRecord](human_review_record.example.json) | queue ID, evidence IDs, and review-context fingerprint |
| Release | [ReleaseGateResult](release_gate_result.example.json) and [ReleasedAudioArtifact](released_audio_artifact.example.json) | current review ID and gate ID |

The values represent a clean illustrative path: machine QA PASS, no pronunciation review, a fictional APPROVE record, and a RELEASED gate result. In actual operation, QA evidence must come from the audio, a person must review the exact artifact, and the release gate recomputes freshness and policy checks. A model-generated WAV alone has no release authority.
