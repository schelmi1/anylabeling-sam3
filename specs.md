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
./webapp/backend/run.sh
```

Open in browser:

```text
http://127.0.0.1:8000
```

Optional host/port overrides:

```bash
ANYLABELING_WEBAPP_HOST=0.0.0.0 ANYLABELING_WEBAPP_PORT=8000 ./webapp/backend/run.sh
```

Remote usage via SSH port-forward:

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<remote-host>
```

## Open Items / Pending Clarifications

- Persistence format and location for browser-side annotations beyond manual JSON export.
- Folder-based dataset navigation and keyboard-heavy annotation workflows in browser app.
- SAM3 memory reduction strategy beyond runtime options (e.g., fp16 model assets, reduced input exports).

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
