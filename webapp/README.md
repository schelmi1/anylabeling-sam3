# AnyLabeling Browser App (SAM2/SAM3)

This adds a lightweight browser UI alongside the existing desktop GUI.

- Desktop GUI remains unchanged (`anylabeling` command).
- Browser app runs locally (or via SSH port-forward).

## Layout

- `webapp/backend/app.py`: FastAPI server + SAM2/SAM3 inference endpoints
- `webapp/frontend/index.html`: single-page browser client

## Run

From repo root:

```bash
pip install -r webapp/backend/requirements.txt
pip install -e . --no-deps
./webapp/backend/run.sh
```

Open: `http://127.0.0.1:8000`

## Remote GPU over SSH

On remote machine:

```bash
./webapp/backend/run.sh
```

On local machine:

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<remote-host>
```

Then browse to `http://127.0.0.1:8000` locally.

## API

- `GET /api/models`
- `POST /api/model/load`
- `POST /api/image/upload` (multipart file)
- `POST /api/infer`

## Notes

- Models are resolved from `~/anylabeling_data/models/<model_name>/`.
- If missing, backend will download/extract SAM2/SAM3 model assets using model config download URLs.
- Current UI is intentionally minimal (point+/point-/rectangle prompts, polygon overlay).
