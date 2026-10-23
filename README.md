# BitPredictor AI — Python Backend

FastAPI + PyTorch backend for model training, prediction, and evaluation.
The React frontend calls this service over REST.

## Run locally (optional, for testing)

```bash
cd python_backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then in the app's **Training** tab, set the backend URL to `http://localhost:8000` and click **Test**.

## Deploy to Render (recommended, free tier, no credit card)

Render hosts a public URL for your backend that the frontend can call.

### Step 1 — Put this folder on GitHub
1. Create a new GitHub repository (e.g. `bitpredictor-backend`).
2. Upload **the contents of this `python_backend` folder** to the root of that repo
   (so `requirements.txt`, `main.py`, etc. are at the top level). You can drag-and-drop
   files on GitHub's "uploading files" page, or use git.

### Step 2 — Create the service on Render
1. Sign up at [render.com](https://render.com) (free, no card needed).
2. Click **New +** → **Web Service**.
3. Connect your GitHub account and select the repository from Step 1.
4. Fill in:
   - **Name**: `bitpredictor-backend`
   - **Runtime**: `Python 3`
   - **Start Command**:
     ```
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan**: `Free`
5. Click **Create Web Service**.

Render installs dependencies and starts the service. Wait for the status to show
**Live** (first deploy takes ~2–4 minutes).

### Step 3 — Copy the URL
At the top-left of your service page you'll see a URL like
`https://bitpredictor-backend.onrender.com`. Copy it.

### Step 4 — Connect the app
1. Open BitPredictor AI and go to the **Training** tab.
2. Paste the URL into **Python Backend URL**.
3. Click **Test** — it should show **Connected**.
4. Train as usual; everything now runs on PyTorch in Python.

> **Note on the free tier:** the service sleeps after 15 minutes of inactivity and
> takes ~30 seconds to wake on the first request after sleep. The "Test" button handles
> this — just click it again if the first request is slow.

## Deploy elsewhere

A `Dockerfile` is included, so any container host works (Railway, Fly.io, a VPS, etc.):
```bash
docker build -t bitpredictor-backend .
docker run -p 8000:8000 -e PORT=8000 bitpredictor-backend
```

## Endpoints

| Method | Path       | Description                                              |
|--------|------------|---------------------------------------------------------|
| GET    | `/health`  | Health check (`{"status":"ok"}`)                        |
| POST   | `/train`   | Streams NDJSON: per-epoch progress, then final results   |
| POST   | `/predict` | Runs a trained model (weights + recent data) → `{prob, direction}` |