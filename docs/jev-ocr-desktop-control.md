# Jev OCR desktop control

Adam can inspect the screen with OCR and use TypeSafe Jev 1.13 over OpenRouter to choose the next operation and OCR text target. Screenshot pixels remain inside Adam for OCR and are not attached to the conversation/model input. Adam's conversational model orchestrates larger tasks across system, browser, and desktop tools; Jev selects bounded actions and targets for visible text-labelled UI. This is a proof of concept, not a general image detector.

## Configure Jev

Install the OCR extra alongside your existing runtime extras (choose the runtime extra you already use; do not enable both CPU and NVIDIA):

```sh
uv sync --extra runtime-nvidia --extra computer-ocr --extra browser-control --extra speaker-verification --extra intent-routing --extra nemotron-diarization
```

Adam uses RapidOCR with PP-OCRv6 medium for text detection and recognition. The setup wizard downloads its approximately 133 MB of model weights so the first desktop observation does not trigger a model download. Inference currently runs on CPU.

Jev is called through OpenRouter's Decisions API with model `typesafe/jev-1.13`. Configure an OpenRouter key in `llm.api_key` or `OPENROUTER_API_KEY`; no local decision-model service or GPU is required.

## Adam configuration

`config.yaml` enables OCR-only observations and Jev's bounded desktop operation/target loop. Visible UI tasks go through `desktop_task`; Adam can combine that with app launch, browser, shell, and read-only system tools for general desktop work. Jev may abstain or reject a low-confidence/unknown region, in which case Adam makes no click.

The action flow is:

1. Capture the focused application internally and extract OCR text boxes.
2. Send the user goal and OCR regions to Jev for one operation and target choice.
3. Convert Jev's selected text region to a coordinate internally, validate the snapshot and active-window bounds, execute one action, then OCR the new state.
4. Ask Jev a separate Yes/No completion question with the full fresh OCR page. On No, request the next operation and repeat. On Yes, return `COMPLETION_CANDIDATE`; on an unclear answer, stop for review.
5. Return an explicit task status and compact OCR excerpt. Adam must re-observe after `INCOMPLETE` and independently verify any `COMPLETION_CANDIDATE`. Explicitly quoted targets must match OCR text before a click is dispatched.

This works across visible desktop apps, not only browser pages. Adam's conversational model remains responsible for planning, opening apps, system commands, and deciding whether the observed state meets the user's goal. OCR cannot identify icon-only controls by itself; AT-SPI and optional screenshot grounding can cover controls that have no readable label.
