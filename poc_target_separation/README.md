# Target separation proof of concept

This standalone runner was used to evaluate target-speaker separation before it
was integrated. Adam's runtime implementation now lives in
`src/stt/target_separator.py`; meeting mode continues to transcribe unfiltered
mixed audio and only uses diarization to assign speaker labels.

Run it from the repository root with:

```sh
PYTHONPATH=. .venv/bin/python poc_target_separation/run_sepformer_poc.py
```

The first run downloads the SpeechBrain model into `speechbrain-model/`. WAVs
and `report.json` are written under `outputs/`. These local outputs can contain
private audio and transcripts and are excluded from Git. `source-0.wav` and
`source-1.wav` are raw separator outputs; `target.wav` is created only when one
source matches the enrolled voice profile.

This is a two-source separator trained on Libri2Mix, so its results on room
microphone audio with video playback are experimental. SpeechBrain notes that
it does not guarantee performance on other datasets.
