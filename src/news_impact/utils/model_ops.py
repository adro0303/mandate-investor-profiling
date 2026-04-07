from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from news_impact.modeling import MLPRegressor


def train_target_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    model_params: dict,
) -> dict:
    if model_name in {"baseline_zero", "baseline_last"}:
        return {"kind": "baseline"}

    if model_name == "knn":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=int(model_params.get("knn_k", 15)))),
            ]
        )
        pipeline.fit(X_train, y_train)
        return {"kind": "sklearn", "model": pipeline}

    if model_name == "ridge":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(model_params.get("ridge_alpha", 1.0)))),
            ]
        )
        pipeline.fit(X_train, y_train)
        return {"kind": "sklearn", "model": pipeline}

    if model_name == "rf":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(model_params.get("rf_estimators", 120)),
                        random_state=int(model_params.get("random_state", 42)),
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        return {"kind": "sklearn", "model": pipeline}

    if model_name == "mlp":
        if X_val is None or y_val is None:
            raise ValueError("MLP requires X_val/y_val for early stopping.")

        device = torch.device(model_params.get("device", "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_t = scaler.fit_transform(imputer.fit_transform(X_train))
        X_val_t = scaler.transform(imputer.transform(X_val))

        train_ds = TensorDataset(
            torch.tensor(X_train_t, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_t, dtype=torch.float32),
            torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32),
        )

        train_dl = DataLoader(train_ds, batch_size=int(model_params.get("batch_size", 256)), shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=int(model_params.get("batch_size", 256)), shuffle=False)

        model = MLPRegressor(
            in_dim=X_train_t.shape[1],
            out_dim=1,
            hidden=int(model_params.get("hidden", 256)),
            dropout=float(model_params.get("dropout", 0.1)),
        ).to(device)
        loss_fn = nn.MSELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=float(model_params.get("lr", 1e-3)))

        best_state = None
        best_val = float("inf")
        patience = int(model_params.get("patience", 10))
        patience_left = patience
        epochs = int(model_params.get("epochs", 30))

        for _ in range(epochs):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            model.eval()
            vals = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    vals.append(float(loss_fn(model(xb), yb).item()))
            val_loss = float(np.mean(vals)) if vals else float("inf")

            if val_loss < best_val:
                best_val = val_loss
                patience_left = patience
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            "kind": "mlp",
            "model": model.cpu(),
            "imputer": imputer,
            "scaler": scaler,
            "best_val_mse": best_val,
        }

    raise ValueError(f"Unsupported model: {model_name}")


def predict_target_model(
    model_name: str,
    trained: dict,
    X_test: np.ndarray,
    y_prev: np.ndarray | None = None,
) -> np.ndarray:
    if model_name == "baseline_zero":
        return np.zeros(X_test.shape[0], dtype=float)
    if model_name == "baseline_last":
        if y_prev is None:
            return np.zeros(X_test.shape[0], dtype=float)
        out = np.array(y_prev, dtype=float)
        out[~np.isfinite(out)] = 0.0
        return out
    if trained["kind"] == "sklearn":
        return trained["model"].predict(X_test).reshape(-1)
    if trained["kind"] == "mlp":
        x_t = trained["scaler"].transform(trained["imputer"].transform(X_test))
        model = trained["model"]
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(x_t, dtype=torch.float32)).numpy().reshape(-1)
        return pred
    raise ValueError("Could not predict with the given artifact.")


def save_target_artifact(target_path: Path, model_name: str, trained: dict) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if model_name in {"baseline_zero", "baseline_last"}:
        return
    if trained["kind"] == "sklearn":
        joblib.dump(trained["model"], target_path.with_suffix(".joblib"))
        return
    if trained["kind"] == "mlp":
        torch.save(trained["model"].state_dict(), target_path.with_suffix(".pt"))
        joblib.dump(
            {"imputer": trained["imputer"], "scaler": trained["scaler"]},
            target_path.with_name(target_path.name + "_prep.joblib"),
        )
        return
    raise ValueError("Unknown artifact type.")


def load_target_artifact(target_path: Path, model_name: str, model_params: dict, n_features: int) -> dict:
    if model_name in {"baseline_zero", "baseline_last"}:
        return {"kind": "baseline"}
    if model_name in {"knn", "ridge", "rf"}:
        return {"kind": "sklearn", "model": joblib.load(target_path.with_suffix(".joblib"))}
    if model_name == "mlp":
        prep = joblib.load(target_path.with_name(target_path.name + "_prep.joblib"))
        model = MLPRegressor(
            in_dim=n_features,
            out_dim=1,
            hidden=int(model_params.get("hidden", 256)),
            dropout=float(model_params.get("dropout", 0.1)),
        )
        state = torch.load(target_path.with_suffix(".pt"), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return {"kind": "mlp", "model": model, "imputer": prep["imputer"], "scaler": prep["scaler"]}
    raise ValueError(f"Unsupported model for loading: {model_name}")


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    rt = rankdata(y_true)
    rp = rankdata(y_pred)
    return float(np.corrcoef(rt, rp)[0, 1])
