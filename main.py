"""FastAPI backend for BitPredictor AI.

Endpoints:
  GET  /health   -> service health check
  POST /train    -> streams per-epoch progress (NDJSON), final line holds results
  POST /predict  -> runs a trained model on recent data
"""
import json
import time
import random

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models import create_model, weights_to_dict, model_from_weights
from data import prepare_data, compute_features, apply_scaler, compute_target_series
from metrics import compute_metrics, compute_regression_metrics

app = FastAPI(title="BitPredictor AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "torch": torch.__version__}


def _train_stream(payload):
    config = payload["config"]
    torch.manual_seed(int(config["seed"]))

    (X_train, y_train, X_test, y_test, feature_names,
     scaler, scaler_params, test_dates) = prepare_data(
        payload["data"], payload["indicators"], config)

    input_size = len(feature_names)
    model = create_model(config, input_size)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["learningRate"]))
    is_reg = config.get("taskMode") == "regression"
    loss_fn = torch.nn.MSELoss() if is_reg else torch.nn.BCELoss()

    Xtr = torch.tensor(X_train)
    ytr = torch.tensor(y_train)
    Xte = torch.tensor(X_test)
    yte = torch.tensor(y_test)

    epochs = int(config["epochs"])
    bs = int(config["batchSize"])
    history = {"loss": [], "valLoss": [], "acc": [], "valAcc": []}
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = loss_fn(out, ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_out = model(Xtr)
            tr_loss = loss_fn(tr_out, ytr).item()
            te_out = model(Xte)
            te_loss = loss_fn(te_out, yte).item()
            if is_reg:
                tr_acc = (torch.sign(tr_out) == torch.sign(ytr)).float().mean().item()
                te_acc = (torch.sign(te_out) == torch.sign(yte)).float().mean().item()
            else:
                tr_acc = ((tr_out >= 0.5).float() == ytr).float().mean().item()
                te_acc = ((te_out >= 0.5).float() == yte).float().mean().item()
        history["loss"].append(tr_loss)
        history["valLoss"].append(te_loss)
        history["acc"].append(tr_acc)
        history["valAcc"].append(te_acc)

        yield json.dumps({
            "type": "epoch", "epoch": epoch + 1, "total": epochs,
            "loss": tr_loss, "valLoss": te_loss, "acc": tr_acc, "valAcc": te_acc,
        }) + "\n"

    model.eval()
    with torch.no_grad():
        te_out = model(Xte).cpu().numpy()
        tr_out = model(Xtr).cpu().numpy()

    y_train_arr = np.asarray(y_train, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)
    weights = weights_to_dict(model, config, input_size)
    duration = time.time() - t0

    if is_reg:
        reg_train = compute_regression_metrics(y_train_arr, tr_out)
        reg_test = compute_regression_metrics(y_test_arr, te_out)
        y_true_tr = (y_train_arr > 0).astype(int)
        y_true_te = (y_test_arr > 0).astype(int)
        train_metrics = compute_metrics(y_true_tr, tr_out, threshold=0.0, allow_brier=False)
        test_metrics = compute_metrics(y_true_te, te_out, threshold=0.0, allow_brier=False)
        metrics_out = {"train": train_metrics, "test": test_metrics,
                       "regression": {"train": reg_train, "test": reg_test}}
        predictions = [{
            "date": test_dates[i],
            "actual": int(y_true_te[i]),
            "prob": float(te_out[i]),
            "predicted": int(te_out[i] >= 0),
            "actualReturn": float(y_test_arr[i]),
            "predictedReturn": float(te_out[i]),
        } for i in range(len(y_test))]
    else:
        train_metrics = compute_metrics(y_train_arr.tolist(), tr_out.tolist())
        test_metrics = compute_metrics(y_test_arr.tolist(), te_out.tolist())
        metrics_out = {"train": train_metrics, "test": test_metrics}
        predictions = [{
            "date": test_dates[i],
            "actual": int(y_test[i]),
            "prob": float(te_out[i]),
            "predicted": 1 if te_out[i] >= 0.5 else 0,
        } for i in range(len(y_test))]

    yield json.dumps({
        "type": "done",
        "weights": weights,
        "metrics": metrics_out,
        "history": history,
        "predictions": predictions,
        "scaler_params": scaler_params,
        "feature_names": feature_names,
        "duration_seconds": round(duration, 2),
    }) + "\n"


@app.post("/train")
def train(payload: dict):
    return StreamingResponse(_train_stream(payload), media_type="application/x-ndjson")


@app.post("/predict")
def predict(payload: dict):
    weights = payload["weights"]
    config = weights["config"]
    model = model_from_weights(weights)

    import pandas as pd
    df = pd.DataFrame(payload["data"])
    if "dates" in df.columns:
        df = df.rename(columns={"dates": "date"})
    feats, _ = compute_features(df, payload["indicators"], config.get("transforms", []))
    raw = feats.dropna().values.astype(np.float32)

    seq_len = int(config["sequenceLength"])
    seq = raw[-seq_len:]
    scaled = apply_scaler(seq, payload["scaler_params"]).astype(np.float32)
    x = torch.tensor(scaled[None])

    with torch.no_grad():
        out = float(model(x).item())
    is_reg = config.get("taskMode") == "regression"
    if is_reg:
        target_type = config.get("targetType", "return")
        T = compute_target_series(df["close"], target_type, config.get("customTargetCode"))
        last_T = float(T.dropna().iloc[-1]) if not T.dropna().empty else float(df["close"].iloc[-1])
        if target_type in ("return", "log_return"):
            direction = "UP" if out > 0 else "DOWN"
        else:
            direction = "UP" if out > last_T else "DOWN"
        return {"prob": out, "predictedValue": out, "direction": direction}
    return {"prob": out, "direction": "UP" if out >= 0.5 else "DOWN"}


def _frame(data):
    import pandas as pd
    df = pd.DataFrame(data)
    if "dates" in df.columns:
        df = df.rename(columns={"dates": "date"})
    return df


@app.post("/talib")
def talib_indicators(payload: dict):
    try:
        from talib_features import compute_all_talib
        ind, groups, skipped = compute_all_talib(_frame(payload["data"]))
    except ImportError:
        raise HTTPException(status_code=500, detail="TA-Lib is not installed on the backend (pip install TA-Lib).")
    columns = {
        "TA_" + c: [float(v) if np.isfinite(v) else None for v in ind[c].to_numpy()]
        for c in ind.columns
    }
    return {
        "columns": columns,
        "groups": {"TA_" + c: g for c, g in groups.items()},
        "skipped": skipped,
    }


@app.post("/cluster")
def cluster(payload: dict):
    from clustering import cluster_features
    try:
        return cluster_features(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


SEARCH_SPACE = {
    'hiddenUnits': [8, 16, 32, 64],
    'numLayers': [1, 2, 3],
    'dropoutRate': [0.0, 0.1, 0.2, 0.3],
    'learningRate': [0.0005, 0.001, 0.002, 0.005],
    'batchSize': [16, 32, 64],
    'sequenceLength': [20, 30, 45, 60],
}

MLP_SEARCH = {
    'mlpNeurons': [32, 64, 128],
    'mlpLayers': [1, 2, 3],
    'mlpActivation': ['relu', 'tanh'],
    'dropoutRate': [0.0, 0.1, 0.2, 0.3],
    'learningRate': [0.0005, 0.001, 0.002, 0.005],
    'batchSize': [16, 32, 64],
    'sequenceLength': [20, 30, 45, 60],
}


def _train_eval(config, data, indicators, epochs):
    torch.manual_seed(int(config.get("seed", 42)))
    (X_train, y_train, X_test, y_test, feature_names,
     scaler, scaler_params, test_dates) = prepare_data(data, indicators, config)
    input_size = len(feature_names)
    model = create_model(config, input_size)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["learningRate"]))
    is_reg = config.get("taskMode") == "regression"
    loss_fn = torch.nn.MSELoss() if is_reg else torch.nn.BCELoss()
    Xtr = torch.tensor(X_train)
    ytr = torch.tensor(y_train)
    Xte = torch.tensor(X_test)
    yte = torch.tensor(y_test)
    bs = int(config["batchSize"])
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for j in range(0, len(Xtr), bs):
            idx = perm[j:j + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        te_out = model(Xte).cpu().numpy()
    y_test_arr = np.asarray(y_test, dtype=float)
    if is_reg:
        return compute_regression_metrics(y_test_arr, te_out)
    return compute_metrics(y_test_arr.tolist(), te_out.tolist())


def _score(metrics, metric, is_reg):
    if is_reg:
        if metric == 'r2':
            return metrics.get('r2', 0.0)
        if metric == 'rmse':
            return -metrics.get('rmse', 0.0)
        return metrics.get('directionalAccuracy', 0.0)
    return metrics.get(metric, metrics.get('accuracy', 0.0))


def _optimize_stream(payload):
    config = payload["config"]
    n_trials = int(payload.get("n_trials", 10))
    epochs_per_trial = int(payload.get("epochs_per_trial", 15))
    metric = payload.get("metric", "accuracy")
    data = payload["data"]
    indicators = payload["indicators"]
    is_reg = config.get("taskMode") == "regression"
    space = {}
    requested = payload.get("search") or {}
    for k, vals in requested.items():
        if vals:
            space[k] = vals
    if not space:
        space = MLP_SEARCH if config["modelType"] == "MLP" else SEARCH_SPACE
    rng = random.Random(int(config.get("seed", 42)))
    best = None
    trials = []
    t0 = time.time()
    for i in range(n_trials):
        trial_config = dict(config)
        sampled = {}
        for k, vals in space.items():
            v = rng.choice(vals)
            trial_config[k] = v
            sampled[k] = v
        try:
            metrics = _train_eval(trial_config, data, indicators, epochs_per_trial)
            score = float(_score(metrics, metric, is_reg))
            trial = {"index": i, "params": sampled, "score": score, "metrics": metrics}
        except Exception as e:
            trial = {"index": i, "params": sampled, "score": None, "error": str(e)}
            score = None
        trials.append(trial)
        if score is not None and (best is None or score > best["score"]):
            best = trial
        yield json.dumps({"type": "trial", "trial": trial, "best": best}) + "\n"
    yield json.dumps({
        "type": "done",
        "trials": trials,
        "best": best,
        "metric": metric,
        "duration_seconds": round(time.time() - t0, 2),
    }) + "\n"


@app.post("/optimize")
def optimize(payload: dict):
    return StreamingResponse(_optimize_stream(payload), media_type="application/x-ndjson")