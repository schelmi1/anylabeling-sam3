# AnyLabeling-SAM3 Working Specs

This is a living specification for the project. It records requirements provided by the user and is updated continuously.

## Meta Requirement

- Always update this `specs.md` whenever the user gives new requirements or changes existing ones.

## Scope

- Keep the existing desktop GUI (`anylabeling`) in place.
- Add a separate browser-based app that can run on localhost or via SSH port-forwarded localhost.
- Focus model support on SAM2 and SAM3 for the browser app.

## Existing Desktop App Requirements

- Desktop app remains functional and unchanged as primary workflow.
- CUDA behavior and stability improvements are acceptable when they do not remove desktop functionality.

## Browser App Requirements

### Architecture

- Browser app lives separately from desktop GUI code path.
- Single-user localhost style usage (not full multi-tenant SaaS web app).
- Support remote usage through SSH port forwarding.
- Add dataset-folder workflow: user can provide/select a folder path to load image + LabelMe JSON pairs.

### Core Inference

- Support SAM2 and SAM3 inference.
- Remove separate "Infer Crop" button; provide one "Infer Text" button that runs on crop when crop is selected, otherwise on full image.
- Image upload and interactive prompting in browser UI.
- Prompt types:
  - Positive point
  - Negative point
  - Rectangle
- SAM3 text prompt support.
- For SAM3, user must be able to choose prompt mode: text only, visual only, or both.

### Model Families

- Introduce a new `metaclip2` model family using Hugging Face Transformers implementation.
- For now, support only:
  - `facebook/metaclip-2-worldwide-s16`
  - `facebook/metaclip-2-worldwide-s16-384`
- `metaclip2` models should be visible in the browser app model dropdown.
- When a `metaclip2` model is selected, the UI should expose a text-label prompt input (comma-separated labels) and show ranked image-vs-text probabilities.
- Add a MetaCLIP minimum probability input (`0..1` float) in browser UI.
- Add a gallery-level batch inference action that runs MetaCLIP over all loaded dataset images.
- Batch inference must write top label and top probability into LabelMe top-level `flags` for each image JSON.
- Gallery cards must display the stored top label and probability for each item.
- Batch inference must also persist all prompt labels with their probabilities into LabelMe `flags`.
- Gallery thumbnails must overlay the full per-label probability list for each image.
- During gallery batch inference, show a responsive progress bar in the UI.
- Gallery overlay should display only the top 3 labels/probabilities per image.
- For MetaCLIP label prompts, automatically template each user label as `an image showing <label>` before model scoring.
- Introduce a `blip2` model family using Hugging Face Transformers implementation.
- For now, support:
  - `Salesforce/blip2-opt-2.7b`
  - `Salesforce/blip2-opt-2.7b-coco`
- `blip2` models should appear in the same browser classifier workflow as MetaCLIP2 (text-label scoring, top label/probabilities, gallery batch infer).
- When BLIP2 is selected and gallery batch infer runs, each image JSON `flags` must include `blip2tag: <top_label>`.
- If image-level label flags exist in paired JSON files, UI must allow filtering the gallery by those flags.
- Gallery flag filtering must support an additional probability threshold (`0..1`) for label-based filtering.
- Gallery stats should be gallery-wide (not per-image): count top-3 labels from each image flags, show label distribution in a hoverable chart, and allow switching chart mode between pie and bar in the left sidebar.
- Clicking the gallery-wide chart should toggle diagram style between pie and bar (and keep selector in sync).
- Chart style is controlled exclusively by clicking the gallery-wide chart (no separate dropdown control).
- When a specific gallery label filter is selected, threshold must apply to that selected label's probability for each image (e.g. `aircraft >= 0.5`).
- When no specific gallery label filter is selected, threshold applies to top-label probability.
- Gallery label threshold control should be a slider with live numeric readout.
- In gallery cards, keep a compact top-1 label tag; when more labels exist, append `...` and show the full label/probability list on hover of that lower tag.
- Gallery image preview should also show top-3 labels/probabilities as an in-image overlay.
- Sidebar order requirement: place Annotations section above Controls section.
- Controls section height should be content-driven/dynamic (no fixed/stretch row allocation).
- Annotation list panel should be vertically resizable by the user.
- `Clear Prompts` and `Clear Masks` controls should live in the Prompt Tools section (not Controls).
- `Run Whole Gallery` should execute only on the currently filtered gallery subset (respecting active gallery label filter and threshold), not on all loaded images.
- Main stage should clearly communicate sections as tabs: `Gallery` and `Image`; the prompt instruction text should appear in the `Image` tab header.
- Model dropdown should be categorized: classifier families (BLIP2/MetaCLIP2) under `Gallery Models`, SAM models under `Image Models`.
- Gallery classifier batch API/UX naming should be model-family agnostic (`classifier`), not MetaCLIP-specific, when BLIP2 is selected.
- If a gallery model (BLIP2/MetaCLIP2) is selected, Prompt Tools must be visually grayed out and disabled.
- Gallery model batch processing progress UI should appear centered in the main stage and show detailed live info (selected model, current input size, current item, dynamic progress bar).
- When BLIP2 is selected, classifier label textbox should be hidden and BLIP2 should generate captions; gallery writes caption into flags via `blip2tag`.
- BLIP2 should run a second-stage KeyBERT keyword extraction on the generated caption and persist these label keywords to `flags.blip2_keywords`; these keywords are the effective BLIP2 labels for gallery workflows.
- Center progress HUD should be larger for readability, place current item on the left, and become clickable to close after run completion/failure.
- Gallery running/progress UI should render as a separate top-layer overlay element above the whole app while running.
- Progress overlay should display initialization stages (e.g., loading model weights / preparing pipeline) before per-image progress begins.
- Gallery should provide a label-source selector so users can choose whether gallery labels/filtering/stats are driven by BLIP2 labels or MetaCLIP2 labels.
- Header should include a `?` help control that launches a guided tutorial tour: step-by-step, highlighting relevant elements, describing their purpose, and requiring explicit user continue action for each step.
- Gallery filtering should support a `First word only` mode: filter options display first words only, and matching/threshold checks should be performed against first-word-normalized labels.
- Gallery filter matching must use labels from the currently selected gallery label source (to avoid unrelated-flag mismatches), with fallback to raw image flags only when no classifier labels exist.
- Guided tutorial content should explicitly explain model categories: Gallery Models perform dataset-wide classification/tagging runs, while Image Models perform per-image interactive inference.
- Guided tutorial should include an explicit `Exit` button so users can stop the tour at any step.
- Gallery filter dropdown should display per-label image counts and sort labels by descending count.
- Add a gallery filter checkbox to sort image-flag labels alphabetically (A→Z) as an alternative to count-based sorting.
- Application branding for the web app should be `AnyLabeling-Next`.
- If an Image Model is selected, `Run Whole Gallery` must be disabled/greyed out; it is enabled only for Gallery Models.
- If a SAM2 model is selected, the SAM3 text prompt textbox should be disabled/greyed out.
- Disabled `Run Whole Gallery` button state should be strongly visually greyed out for clear affordance.
- `Classifier Min Probability` input should be disabled/greyed out when an Image Model is selected.
- `Confidence` input should apply to both SAM2 and SAM3 (and only be disabled for Gallery Models).

### GPU / Runtime Controls

- Must be able to unload models and clear GPU memory from UI.
- VRAM controls panel is the unified controls panel (not a separate VRAM-only section).
- VRAM-related runtime options must be user-configurable via checkboxes.
- VRAM options use an explicit `Apply` button.
- VRAM options default to ON.
- Applying VRAM options should unload currently loaded models so settings take effect.
- Controls panel must include `Clear Prompts` and `Clear Masks` actions.
- Controls panel must include `Autosave JSON` checkbox, enabled by default.
- Autosave JSON should persist annotations after each annotation workflow step (save/edit/delete/clear/reset).
- Autosave JSON should be written as a sidecar file in the same folder as the stored uploaded image copy (image+json pair).
- Sidecar JSON filename must use the exact same filestem as the image (e.g. `image.png` -> `image.json`).
- If `Autosave JSON` is enabled, `Clear Masks` must show a warning confirmation that shapes in sidecar JSON will be deleted/overwritten.
- Controls panel must include a configurable `Working Directory` used by the browser app for stored images and sidecar JSON pairs.

Current VRAM options:

- CUDA low-memory mode
- Disable TensorRT
- Cap CUDA memory
- Tiny embedding cache
- Disable preload

### Annotation UX

- Browser app must include an annotations list panel.
- Annotations must be nameable/editable.
- Different label names must render with different annotation colors (consistent per label across the session).
- Annotation list rows/inputs should visually use the corresponding label color.
- Use a distinctive, high-contrast categorical color palette for label colors.
- Annotation entries should be selectable and deletable.
- Clicking an annotation in the list must select it and visibly highlight its geometry on canvas.
- Hovering an annotation should show its label and slightly highlight its polyline.
- Hovering an annotation row in the annotations list must immediately highlight the corresponding polyline on canvas.
- Hovering over annotation regions on the image canvas must also show label and slight highlight.
- Label must be shown near the mouse cursor when hovering an annotation region.
- Annotation export should be available (JSON).
- When an image is loaded, if a corresponding sidecar JSON exists in the working directory image folder, annotations must be auto-loaded into the annotation list and canvas.
- When loading from dataset folder, if LabelMe JSON with matching stem exists, annotations must auto-load from that JSON.
- After inference creates mask(s), app must prompt for a label name before saving annotations.
- Replace modal prompt with a multi-selection radial wheel at current mouse position.
- Wheel should open outside the mask proposal bounding box (next to proposal), not on top of cursor/proposal.
- Remove manual full-image run button; when point or rectangle prompt is placed, run inference automatically and open wheel on proposal.
- Wheel must support: typing label name, selecting existing labels, saving, and cancel/discard actions.
- If user saves using an existing label, that existing label should be pre-selected in subsequent wheel prompts (when still available).
- Wheel must include a "Same label" option so multi-mask saves can use one identical label instead of instance suffixes.
- While wheel is open, user must be able to drag/move it aside on screen.
- Wheel action buttons should appear in front of the center label panel; existing-label action text should read "Save Existing".
- Wheel action buttons must be placed outside the wheel ring, not overlapping the center panel.
- Label prompt must open only after mask proposal is visible to the user.
- No intermediate save-proposal button: flow must be inference -> proposal visible -> prompt.
- Saving from that prompt should create named annotations (and support multiple masks).
- When annotations are saved, current prompt marks must be cleared automatically.
- After saving through the wheel, UI must instantly return to a no-prompt-active ready state (wheel closed, proposal cleared) while retaining the last active prompt tool.
- When discarding from the wheel, UI must also return to no-prompt-active ready state with prompts cleared and default prompt mode active.

### Drawing UX

- Rectangle drawing must show live preview while dragging, not only after mouse release.
- Prompt tool buttons must visually indicate which tool is currently active.
- Provide a crop tool to select a region from the full image.
- Show crop preview next to the full image.
- Crop preview panel should sit close to the actual rendered image area (avoid large empty gap from oversized container).
- Crop preview must support upscaling for easier prompting.
- Crop preview must show source crop resolution and effective upscaled resolution.
- Must support running inference on the crop.
- For SAM3 crop inference: if mark-guided crop inference returns no masks, retry text-only fallback automatically.
- Crop inference outputs must be mapped back to full-image coordinates.
- During crop inference, proposals must also be visible in the crop preview.
- Crop preview proposals must be rendered as clearly visible dashed highlighted polylines.
- Saved annotations must remain visible as polylines in the crop preview after saving.
- Proposal visuals should be vibrant/high-contrast and dim the non-proposal image region to emphasize proposals.
- During crop inference save, no annotations may be created outside crop bounds.

### UI/UX Direction

- UI should feel modern and not cluttered.
- Group related controls and reduce visual noise.
- Keep advanced controls present but not dominant.
- Show dataset image previews in a gallery strip at the top of the main tab; clicking a preview loads that image and its annotations.
- Keep gallery compact in height (significantly reduced vertical footprint) while preserving thumbnail card width.
- Gallery should not span most of the stage width; keep it as a compact, left-aligned strip.
- Gallery container height should be around 200px.
- Gallery height should be adaptive with ~200px cap, without leaving empty vertical gaps before the main image/crop area.
- Gallery should span full available width across the top of the main tab.
- No large empty vertical gap is allowed between gallery and main workspace content.

## Reliability and Stability Requirements

- Avoid silent CPU fallback when GPU is expected where practical.
- Handle invalid/non-image preload files gracefully.
- Avoid crashes due to background preprocessing errors.

## Operational Requirements

- Run mode for local development and use:
  - backend server + browser frontend served locally.
- Preserve compatibility with existing environment setup.

### Install and Start Server

From repo root:

```bash
pip install -r webapp/backend/requirements.txt
pip install -e . --no-deps
```

Recommended start (GPU-forced browser app):

```bash
./run_anylabeling_gpu.sh
```

This launches the web app server and opens browser automatically at:

```text
http://127.0.0.1:8765
```

Optional host/port overrides for launcher:

```bash
ANYLABELING_WEB_HOST=0.0.0.0 ANYLABELING_WEB_PORT=8765 ./run_anylabeling_gpu.sh
```

Run legacy desktop GTK app explicitly:

```bash
./run_anylabeling_gpu.sh --desktop
```

Alternative direct web server start (without launcher):

```bash
./webapp/backend/run.sh
```

Direct server host/port overrides:

```bash
ANYLABELING_WEBAPP_HOST=0.0.0.0 ANYLABELING_WEBAPP_PORT=8000 ./webapp/backend/run.sh
```

For MetaCLIP2 support, install extras:

```bash
pip install -e ".[metaclip2]"
```

For BLIP2 support, install extras:

```bash
pip install -e ".[blip2]"
```

If MetaCLIP2 load fails with `model type metaclip_2 not recognized`, upgrade transformers in the same runtime env and restart server:

```bash
pip install --upgrade transformers sentencepiece
```

If still failing, install transformers from source:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

### ONNX CUDA Runtime Check (Desktop + Browser)

If ONNX inference still shows CPU, apply shell startup env and verify providers:

```bash
source ~/.bashrc
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
./run_anylabeling_gpu.sh
```

Expected: `CUDAExecutionProvider` appears in provider list, and UI compute chip shows `cuda`.

Quick pre-launch sanity check:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output includes `CUDAExecutionProvider` before launching the app.

If it still falls back to CPU, collect diagnostics:

```bash
python -c "import onnxruntime as ort, sys; print(sys.executable); print(ort.__file__); print(ort.get_available_providers())"
echo "$LD_LIBRARY_PATH"
```

Use these outputs to debug env/package mismatch (wrong interpreter, wrong ORT wheel, or missing CUDA/cuDNN libs).

Remote usage via SSH port-forward:

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<remote-host>
```

## Open Items / Pending Clarifications

- Persistence format and location for browser-side annotations beyond manual JSON export.
- Folder-based dataset navigation and keyboard-heavy annotation workflows in browser app.
- SAM3 memory reduction strategy beyond runtime options (e.g., fp16 model assets, reduced input exports).

## Gallery UX Requirements

- Dataset gallery must support a draggable height resize handle.
- Gallery thumbnails must use `200x200` image previews.
- Gallery layout must auto-manage columns and rows based on available width (responsive grid wrapping).
- Chosen gallery height should persist across reloads.

## Future Requirement Intake (Process)

When a new requirement is provided:

1. Add it under the relevant section above.
2. If it changes behavior, add/update acceptance criteria.
3. If it conflicts with an earlier requirement, mark conflict and preferred resolution.
4. Keep this file as the source-of-truth snapshot.

## Change Log

- 2026-03-25: Initial consolidated spec created from user-provided requirements in this project session.
- 2026-03-25: Added explicit meta-rule to always update `specs.md` and added post-inference label prompt requirement.
- 2026-03-25: Added requirement to auto-clear prompt marks after saving annotations.
- 2026-03-25: Added requirement that label prompt opens only after visible proposal.
- 2026-03-25: Removed save-proposal button requirement in favor of auto prompt after visible proposal.
- 2026-03-25: Added annotation hover-label + polyline highlight requirement.
- 2026-03-25: Added explicit click-to-select annotation highlight requirement.
- 2026-03-25: Added canvas-region hover highlight + label requirement.
- 2026-03-25: Added requirement for cursor-near label on canvas hover.
- 2026-03-25: Added crop-tool workflow requirement (preview, upscale, crop inference, remap).
- 2026-03-25: Added UX requirement for dynamic crop panel proximity to rendered image.
- 2026-03-25: Added SAM3 crop text-only fallback requirement.
- 2026-03-25: Added strict crop-bound save + crop-preview proposal visibility requirement.
- 2026-03-25: Added requirement to display crop effective resolution in preview.
- 2026-03-25: Added requirement for dashed high-contrast proposal polyline in crop preview.
- 2026-03-25: Added SAM3 prompt-mode selector requirement (text/visual/both).
- 2026-03-25: Added radial wheel prompt requirement replacing modal prompt.
- 2026-03-25: Added auto-infer-on-prompt requirement and removed manual run button.
- 2026-03-25: Added requirement for vibrant proposal emphasis with background dimming.
- 2026-03-25: Added draggable wheel requirement.
- 2026-03-25: Added wheel button layering/text requirement (Save Existing).
- 2026-03-25: Added requirement that wheel action buttons sit outside ring.
- 2026-03-25: Added unified Infer Text button requirement (crop-aware).
- 2026-03-25: Added wheel Same label option requirement.
- 2026-03-25: Strengthened requirement that annotation-list row hover must immediately highlight the corresponding canvas polyline.
- 2026-03-25: Added requirement that saved annotations are visible as polylines in crop preview after save.
- 2026-03-25: Converted VRAM section into unified controls panel with clear prompts, clear masks, and autosave JSON (default on).
- 2026-03-25: Changed autosave location to sidecar JSON next to stored uploaded image file.
- 2026-03-25: Added clear-masks warning prompt when autosave is enabled to prevent accidental JSON shape deletion.
- 2026-03-25: Added configurable browser-app working directory support (UI + backend API) for image/json pairing.
- 2026-03-25: Added auto-load of sidecar annotations when loading an image.
- 2026-03-25: Enforced exact same filestem naming for image/json pairs (`<stem>.<img_ext>` + `<stem>.json`).
- 2026-03-25: Added requirement that wheel-save instantly returns to ready mode with no active prompt/proposal state.
- 2026-03-25: Added per-label color mapping requirement so different labels have different annotation colors.
- 2026-03-26: Added requirement that annotation list styling reflects per-label colors.
- 2026-03-26: Upgraded label colors to a more distinctive high-contrast palette.
- 2026-03-26: Added active-state highlight requirement for selected prompt tool button.
- 2026-03-26: Added install and server startup instructions to this specs document.
- 2026-04-01: Added `metaclip2` model family requirement with two allowed HF model IDs.
- 2026-04-01: Added dataset-folder image+LabelMe pair loading and top-of-main-tab gallery preview requirements.
- 2026-04-01: Hardened MetaCLIP2 tokenizer/processor fallback to support non-CLIP tokenizer backends (e.g. XLM-R) in HF models.
- 2026-04-01: Added MetaCLIP2 troubleshooting note for `metaclip_2` architecture requiring newer transformers build.
- 2026-04-01: Reduced gallery vertical size while keeping width unchanged.
- 2026-04-01: Constrained gallery strip width so it does not span most of the screen.
- 2026-04-01: Set gallery height target to about 200px.
- 2026-04-01: Made gallery height adaptive (capped at ~200px) to remove empty gap before workspace.
- 2026-04-01: Set gallery to span full width at the top of the main tab.
- 2026-04-01: Fixed main stage row layout to remove large empty gap between gallery and workspace.
- 2026-04-01: Added draggable gallery height with persistent setting, and switched gallery to responsive `200x200` grid previews with automatic row/column wrapping.
- 2026-04-01: Added MetaCLIP gallery batch inference with configurable min probability, persisted LabelMe `flags` classification fields, and gallery-card display of top label/probability.
- 2026-04-01: Extended MetaCLIP gallery batch inference to persist full label-probability distributions in flags and overlay those scores on each gallery thumbnail.
- 2026-04-01: Added async gallery-inference progress bar, limited gallery overlay to top-3 labels, and added automatic MetaCLIP prompt templating (`an image showing <label>`).
- 2026-04-01: Added gallery filtering by image label flags parsed from JSON `flags`.
- 2026-04-01: Added `blip2` model family support (webapp + desktop registration) with classifier-compatible single-image and gallery inference flow.
- 2026-04-01: Updated model-specific control visibility: show SAM confidence control only for SAM image models.
- 2026-04-01: Updated model-specific control visibility: show classifier minimum probability only for gallery models.
- 2026-04-01: Fixed SAM2 confidence handling for browser/desktop paths by applying confidence threshold in SAM2 mask selection (instead of ignoring it).
- 2026-04-01: Changed SAM confidence control UI from number field to slider with live value display.
- 2026-04-01: Set SAM3 default prompt tool to rectangle when switching into a SAM3 model.
- 2026-04-01: Split combined sidebar card into separate `Session` and `Model` tabs.
- 2026-04-01: Refined tab layout to two separate tab cards (Session card and Model card), switched by shared tab buttons.
- 2026-04-01: Reverted tabbed sidebar layout; Session, Model, Prompt Tools, Annotations, and Controls are standalone cards.
- 2026-04-02: Added dataset folder browse dialog button (context-window style picker) to fill folder path input.
- 2026-04-02: Updated folder picker to prefer desktop-native dialogs (`zenity`/`kdialog`) so it matches normal file chooser style; Tk kept only as fallback.
- 2026-04-02: Fixed dataset loading visibility for image-only/mixed folders by resetting gallery filter/threshold on load and keeping unscored images visible in no-filter mode.
- 2026-04-02: Expanded recognized dataset image extensions (`.jpe`, `.jfif`, `.dib`, `.gif`, `.ppm/.pgm/.pbm/.pnm`) for broader folder compatibility.
- 2026-04-02: Workdir behavior changed: no manual Workdir UI; current workdir is automatically set to the loaded dataset folder.
- 2026-04-02: Dataset image detection now includes OpenCV reader fallback for uncommon image extensions.
- 2026-04-02: Removed `Copy files into workdir` option from UI/flow (workdir follows loaded dataset folder and no copy mode is used).
- 2026-04-02: Added dataset-load diagnostics (`scanned_files`, `image_candidates`) and zero-result status message for easier troubleshooting.
- 2026-04-02: Browse-folder flow now auto-loads selected folder immediately (no extra `Load Folder` click required).
- 2026-04-02: UX/UI pass applied: stronger typography hierarchy and spacing rhythm, reduced border noise, increased active-state contrast, and tighter layout consistency.
- 2026-04-02: Added dynamic color-coded current model-family badge (SAM image model vs gallery classifier).
- 2026-04-02: Enhanced annotations panel with label search, grouping-by-label toggle, and label/annotation counts in summary.
- 2026-04-02: Improved empty-state guidance for annotations and no-proposal inference outcomes with actionable micro-copy.
- 2026-04-02: Added visible shortcut hints and keyboard bindings: `P` point+, `R` rectangle, `S` save proposal (wheel open), `D` discard proposal (wheel open).
- 2026-04-02: Refined proposal control wheel into a compact panel layout (non-radial) for clearer action hierarchy.
- 2026-04-02: Fixed `P`/`R` hotkeys by routing through explicit prompt-tool activation logic with warning feedback when gallery models disable prompt tools.
- 2026-04-02: Reworked status bar from always-green to neutral severity-aware states (info/success/warning/error).
- 2026-04-02: Moved heavy runtime options behind a collapsed `Advanced Controls` bar/button (details panel) to declutter left sidebar default view.
- 2026-04-02: Added cursor-attached prompt badge on canvas to communicate active prompt tool and SAM3 prompt mode near mouse position.
- 2026-04-02: Moved status communication from left-sidebar bottom to inline image-header status chip plus global top-right toast stack.
- 2026-04-02: Added `C` keyboard shortcut for Crop tool (alongside `P`, `R`, `S`, `D`).
- 2026-04-02: Added always-on global top bar (sticky) with auto-scrolling overflow and live chips for loaded folder, loaded model, workspace stats, and latest status.
- 2026-04-02: Added `Model Family` chip to global top bar, synchronized with the sidebar model-family badge state.
- 2026-04-02: Made Session, Model, Prompt Tools, and Annotations cards collapsible/expandable using the same details-summary pattern as Advanced Controls.
- 2026-04-02: Removed duplicate model-family badge from top-left sidebar card; model family is now shown only in the global top bar chip.
- 2026-04-02: Removed manual `Load Model` button; selected dropdown model now auto-loads on selection change (and initial load).
- 2026-04-02: Removed top-right toast status display in favor of global bar status chip + hover history popover.
- 2026-04-02: Added status history interaction: hovering global status chip shows last 10 status messages.
- 2026-04-02: Updated guided tour to match current UX: global top bar, hoverable status history, auto model load, cursor prompt badge, collapsible advanced controls, and updated shortcut guidance.
- 2026-04-02: Model dropdown now defaults to an explicit `Select model...` state with no model loaded at startup; model auto-load triggers only after user selection.
- 2026-04-02: Added explicit model load-state indicator in Model card (`No model loaded` / `Loading` / `Loaded` / `Load failed`) synchronized with selection and unload/reset actions.
- 2026-04-02: Removed duplicate inline status chip from Image header; status is now communicated only through the global top-bar status chip + hover history.
- 2026-04-02: Added global top-bar `Compute` chip showing active compute device and AI runtime, with backend-derived provider details (ONNX EPs or Transformers/PyTorch device) refreshed on startup, model load, runtime apply, and reset.
- 2026-04-02: Re-designed proposal save wheel to a modern floating action card and reduced actions to exactly three buttons: `Save New`, `Save Existing`, and `Discard` (removed `Cancel` action and outside-click dismissal).
- 2026-04-02: Folder browse dialog behavior stabilized: when a native picker (`zenity`/`kdialog`) is available and canceled, do not fall through to Tk fallback; Tk is now used only if native pickers are unavailable or fail.
- 2026-04-02: Added executable GPU launcher script `run_anylabeling_gpu.sh` with forced ONNX CUDA provider defaults (`ANYLABELING_FORCE_CUDA=1`, TensorRT off, CUDA provider priority, `CUDA_VISIBLE_DEVICES=0`); default launch target is browser app (`uvicorn webapp.backend.app:app` with auto-open), with explicit `--desktop` option for legacy GTK app.
- 2026-04-02: Updated startup docs: recommend `./run_anylabeling_gpu.sh` for browser app launch, document `--desktop` fallback, and include launcher/direct host+port override options.
- 2026-04-02: Compute status chip now reports device as `cuda`/`cpu` (instead of `GPU`/`CPU`) and includes active runtime provider details for clearer ONNX CUDA verification.
- 2026-04-02: Added shell-startup configuration guidance and helper block for `~/.bashrc` to auto-extend `LD_LIBRARY_PATH` with pip-installed CUDA/cuDNN library folders used by `onnxruntime-gpu`.
- 2026-04-02: Added explicit ONNX CUDA verification/troubleshooting commands (`source ~/.bashrc`, provider check, launcher run, and diagnostic dump) to startup docs.
- 2026-04-02: Added a compact pre-launch ONNX CUDA sanity-check command to startup documentation.
- 2026-04-02: Crop preview UX improved: larger empty-state card with icon/instruction when no crop is selected; when crop exists, show immediate crop thumbnail plus effective-resolution badge.
- 2026-04-02: Annotations panel made more task-driven: selected-label summary line added, search row made sticky, and annotation list now virtualizes rendering for large lists.
- 2026-04-02: Left-rail spacing/typography rhythm refined: increased inter-card spacing and reduced uppercase micro-label styling (sentence-case labels, lighter letter spacing).
- 2026-04-02: Gallery filters/stats moved out of left sidebar into a dedicated local toolbar directly above gallery thumbnails; `Run Whole Gallery` kept in Model card as a single-row action button.
- 2026-04-02: Finalized gallery-control placement: moved to a dedicated collapsible `Gallery Controls` card in the left rail between `Session` and `Model`; `Run Whole Gallery` remains a single-row button in `Model`.
- 2026-04-02: Moved app identity (`AnyLabeling-Next` + `SAM2/SAM3/MetaCLIP2/BLIP2`) from left sidebar to the left side of the global top status bar, and moved guided-tour `?` button to the top bar to free left-rail space.
- 2026-04-02: Gallery inference progress HUD de-duplicated: avoid repeating processed counters/current item in status line when those are already shown in dedicated fields.
- 2026-04-02: Global top-bar stats annotation count corrected to dataset-wide semantics (sum across dataset items) with live adjustment for the currently open image.
