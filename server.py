from __future__ import annotations

import argparse
import csv
import io
import json
import os
import threading
import time
import uuid
from contextlib import nullcontext
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from encoders.c9_encoder import C9Encoder
from encoders.r9_encoder import R9Encoder
from models.conmismatch9 import ConMismatch9Model
from models.deepfocus import DeepFocusModel
from utils.config import load_config, resolve_dataset_files
from utils.metrics import sigmoid
from utils.sequence import normalize_sequence, risk_level_from_probability
from utils.training import evaluate_model, fit_linear_model, serialize_history, train_test_split_71515
from utils.io import collect_balanced_records


try:  # pragma: no cover - torch is optional for the lightweight fallback server
    import numpy as np
    import torch

    from models.conmismatch9_torch import ConMismatch9TorchConfig, ConMismatch9TorchModel
    from models.deepfocus_torch import DeepFocusTorchConfig, DeepFocusTorchModel

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - keep JSON fallback usable without torch/numpy
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    ConMismatch9TorchConfig = None  # type: ignore[assignment]
    ConMismatch9TorchModel = None  # type: ignore[assignment]
    DeepFocusTorchConfig = None  # type: ignore[assignment]
    DeepFocusTorchModel = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


try:  # pragma: no cover - optional dependency support
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles

    FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency support
    FASTAPI_AVAILABLE = False


class PairInput(BaseModel):
    sgRNA: str
    dna: str


class PredictRequest(BaseModel):
    sgRNA: str
    dna: str
    model: str = Field(default="deepfocus")
    encoder: str = Field(default="r9")


class PredictBatchRequest(BaseModel):
    model: str = Field(default="deepfocus")
    encoder: str = Field(default="r9")
    pairs: list[PairInput]


class TrainRequest(BaseModel):
    config: str | None = None
    config_path: str | None = None
    model: str | None = None
    encoder: str | None = None
    dataset_files: list[str] | None = None
    async_run: bool = True


class PredictFileRequest(BaseModel):
    model: str = Field(default="deepfocus")
    encoder: str = Field(default="r9")
    file_content: str
    format: str = Field(default="csv")
    has_header: bool = Field(default=True)


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _plain_response(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _safe_static_path(request_path: str) -> Path | None:
    """Resolve a /static/ request path safely, preventing directory traversal."""
    if not request_path.startswith("/static/"):
        return None
    relative = request_path[len("/static/"):]
    # URL-decode basic percent-encoding for .. safety
    import urllib.parse
    relative = urllib.parse.unquote(relative)
    # Reject absolute paths
    if relative.startswith("/"):
        return None
    static_root = Path("static").resolve()
    target = (static_root / relative).resolve()
    # target must be inside static_root and must be a file
    try:
        target.relative_to(static_root)
    except ValueError:
        return None
    if not target.exists() or not target.is_file():
        return None
    return target


def _torch_device(requested: str):
    if torch is None:
        raise RuntimeError("torch is not available")
    requested = requested.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class TorchSequencePredictor:
    def __init__(self, model_name: str, encoder_name: str, model: Any, device: str):
        if torch is None or np is None:
            raise RuntimeError("torch and numpy are required for .pt checkpoints")
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.model_backend = f"torch_{model_name}"
        self.device = _torch_device(device)
        self.encoder = R9Encoder() if encoder_name == "r9" else C9Encoder()
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, on_seq: str, off_seq: str) -> dict[str, Any]:
        if torch is None or np is None:
            raise RuntimeError("torch and numpy are required for .pt checkpoints")
        encoded = np.asarray(self.encoder.encode_pair(on_seq, off_seq), dtype=np.float32)
        x = torch.from_numpy(encoded).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            context = torch.amp.autocast(device_type="cuda") if self.device.type == "cuda" else nullcontext()
            with context:
                logit = self.model(x)
            probability = float(torch.sigmoid(logit).detach().cpu().item())
        return {
            "off_target_prob": probability,
            "risk_level": risk_level_from_probability(probability),
            "model_used": self.model_name,
            "encoder_used": self.encoder_name,
            "model_backend": self.model_backend,
        }


def _load_torch_predictor(model_name: str, weights_path: str | Path, device: str) -> TorchSequencePredictor:
    if torch is None or not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available; cannot load .pt checkpoint")
    checkpoint = torch.load(weights_path, map_location="cpu")
    resolved_model = str(checkpoint.get("model_name", model_name)).lower()
    encoder_name = str(checkpoint.get("encoder_name", "r9" if resolved_model == "deepfocus" else "c9")).lower()
    model_config = dict(checkpoint.get("model_config", {}))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    ablation_mode = str(model_config.get("ablation_mode", "full")).lower().replace("-", "_")
    if resolved_model == "conmismatch9" and ablation_mode == "full":
        legacy_keys = (
            "fusion.gate.",
            "fusion.pool_projection.",
            "fusion.head.",
            "fusion.cnn_norm.",
            "fusion.run_norm.",
        )
        if any(any(key.startswith(prefix) for prefix in legacy_keys) for key in state_dict.keys()):
            model_config["ablation_mode"] = "legacy_full"

    if resolved_model == "deepfocus":
        if DeepFocusTorchConfig is None or DeepFocusTorchModel is None:
            raise RuntimeError("DeepFocus torch classes are not available")
        model = DeepFocusTorchModel(DeepFocusTorchConfig(**model_config))
    elif resolved_model == "conmismatch9":
        if ConMismatch9TorchConfig is None or ConMismatch9TorchModel is None:
            raise RuntimeError("ConMismatch9 torch classes are not available")
        model = ConMismatch9TorchModel(ConMismatch9TorchConfig(**model_config))
    else:
        raise ValueError(f"unknown torch model in checkpoint: {resolved_model}")

    model.load_state_dict(state_dict)
    return TorchSequencePredictor(resolved_model, encoder_name, model, device)


class ModelRegistry:
    def __init__(self, config_paths: list[str] | None = None, device: str = "cpu"):
        self.device = device
        self.encoders = {
            "r9": R9Encoder(),
            "c9": C9Encoder(),
        }
        self.models = {
            "deepfocus": DeepFocusModel(),
            "conmismatch9": ConMismatch9Model(),
        }
        self.expected_pairs = {
            "deepfocus": "r9",
            "conmismatch9": "c9",
        }
        self.loaded_from: dict[str, str] = {}
        self.jobs: dict[str, dict[str, Any]] = {}

        if config_paths:
            for config_path in config_paths:
                self._load_configured_weights(config_path)

    def _load_configured_weights(self, config_path: str) -> None:
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            print(f"[server] config not found: {config_path}")
            return
        except Exception as exc:
            print(f"[server] failed to load config {config_path}: {exc}")
            return

        model_name = str(config.get("model", "")).lower()
        weights_path = config.get("weights_path")
        if model_name not in self.models:
            print(f"[server] unknown model in config {config_path}: {model_name}")
            return
        if not weights_path:
            print(f"[server] no weights_path in config {config_path}")
            return

        weights = Path(weights_path)
        if not weights.exists():
            print(f"[server] weights not found for {model_name}: {weights_path}")
            return

        if weights.suffix in {".pt", ".pth"}:
            try:
                self.models[model_name] = _load_torch_predictor(model_name, weights, self.device)
                self.loaded_from[model_name] = str(weights)
                print(f"[server] loaded torch checkpoint for {model_name}: {weights_path}")
            except Exception as exc:
                print(f"[server] failed to load torch checkpoint for {model_name} from {weights_path}: {exc}")
                return
            return

        try:
            model_cls = type(self.models[model_name])
            state = model_cls.load_state(weights)
            self.models[model_name] = model_cls(state=state)
            self.loaded_from[model_name] = str(weights)
            print(f"[server] loaded legacy weights for {model_name}: {weights_path}")
        except Exception as exc:
            print(f"[server] failed to load legacy weights for {model_name} from {weights_path}: {exc}")
            return

    def _model_backend(self, model_name: str) -> str:
        model = self.models[model_name]
        return str(getattr(model, "model_backend", "legacy_json"))

    def health(self) -> dict[str, Any]:
        available = set(self.models.keys())
        loaded = set(self.loaded_from.keys())
        if not loaded:
            status = "degraded"
        elif loaded == available:
            status = "ok"
        else:
            status = "partial"

        fallback_active = any(
            self._model_backend(name) == "legacy_json"
            for name in sorted(self.models.keys())
        )

        return {
            "status": status,
            "fallback_active": fallback_active,
            "available_encoders": list(self.encoders.keys()),
            "available_models": list(self.models.keys()),
            "loaded_models": sorted(loaded),
            "missing_models": sorted(available - loaded),
            "device": self.device,
            "model_backends": {
                name: {
                    "backend": self._model_backend(name),
                    "loaded_from": self.loaded_from.get(name),
                }
                for name in sorted(self.models.keys())
            },
            "torch_available": TORCH_AVAILABLE,
        }

    def predict(self, model_name: str, encoder_name: str, sgRNA: str, dna: str) -> dict[str, Any]:
        model_name = model_name.lower()
        encoder_name = encoder_name.lower()
        if model_name not in self.models:
            raise ValueError(f"unknown model: {model_name}")
        if encoder_name not in self.encoders:
            raise ValueError(f"unknown encoder: {encoder_name}")

        expected = self.expected_pairs.get(model_name)
        if expected and encoder_name != expected:
            raise ValueError(f"model {model_name} expects encoder {expected}, got {encoder_name}")

        from utils.sequence import validate_sequence
        validate_sequence(sgRNA, name="sgRNA", length=23)
        validate_sequence(dna, name="dna", length=23)
        normalized_on = normalize_sequence(sgRNA, 23)
        normalized_off = normalize_sequence(dna, 23)
        model = self.models[model_name]
        payload = model.predict(normalized_on, normalized_off)
        payload["input_seq_len"] = len(normalized_on)
        payload["encoder_used"] = encoder_name
        payload["model_used"] = model_name
        payload["model_backend"] = self._model_backend(model_name)
        return payload

    def predict_batch(self, model_name: str, encoder_name: str, pairs: list[PairInput]) -> list[dict[str, Any]]:
        return [self.predict(model_name, encoder_name, pair.sgRNA, pair.dna) for pair in pairs]

    def list_models(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": name,
                    "encoder": self.expected_pairs.get(name),
                    "backend": self._model_backend(name),
                    "loaded": name in self.loaded_from,
                    "loaded_from": self.loaded_from.get(name),
                }
                for name in sorted(self.models.keys())
            ]
        }

    def predict_from_file(self, model_name: str, encoder_name: str, file_content: str, fmt: str, has_header: bool) -> dict[str, Any]:
        model_name = model_name.lower()
        encoder_name = encoder_name.lower()
        if model_name not in self.models:
            raise ValueError(f"unknown model: {model_name}")
        if encoder_name not in self.encoders:
            raise ValueError(f"unknown encoder: {encoder_name}")
        expected = self.expected_pairs.get(model_name)
        if expected and encoder_name != expected:
            raise ValueError(f"model {model_name} expects encoder {expected}, got {encoder_name}")

        fmt = fmt.lower().strip()
        if fmt not in {"csv", "tsv"}:
            raise ValueError(f"unsupported format: {fmt}; expected csv or tsv")

        delimiter = "\t" if fmt == "tsv" else ","
        rows = list(csv.reader(io.StringIO(file_content.strip()), delimiter=delimiter))
        if not rows:
            raise ValueError("file is empty")

        start_idx = 0
        sgRNA_idx = 0
        dna_idx = 1
        if has_header:
            header = [h.strip().lower() for h in rows[0]]
            try:
                sgRNA_idx = header.index("sgrna")
            except ValueError:
                try:
                    sgRNA_idx = header.index("on")
                except ValueError:
                    sgRNA_idx = 0
            try:
                dna_idx = header.index("dna")
            except ValueError:
                try:
                    dna_idx = header.index("off")
                except ValueError:
                    try:
                        dna_idx = header.index("off-target")
                    except ValueError:
                        dna_idx = 1
            start_idx = 1

        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for i, row in enumerate(rows[start_idx:], start=1):
            if len(row) < max(sgRNA_idx, dna_idx) + 1:
                errors.append({"row": i, "error": f"too few columns (got {len(row)}, need at least {max(sgRNA_idx, dna_idx) + 1})"})
                continue
            sgRNA = row[sgRNA_idx].strip().upper()
            dna = row[dna_idx].strip().upper()
            if not sgRNA:
                errors.append({"row": i, "error": "sgRNA sequence is empty"})
                continue
            if not dna:
                errors.append({"row": i, "error": "dna sequence is empty"})
                continue
            if len(sgRNA) != 23:
                errors.append({"row": i, "error": f"sgRNA length {len(sgRNA)} != 23"})
                continue
            if len(dna) != 23:
                errors.append({"row": i, "error": f"dna length {len(dna)} != 23"})
                continue
            try:
                pred = self.predict(model_name, encoder_name, sgRNA, dna)
                pred["row"] = i
                results.append(pred)
            except Exception as exc:
                errors.append({"row": i, "error": str(exc)})
                continue

        return {
            "results": results,
            "total": len(rows) - start_idx,
            "success": len(results),
            "failed": len(errors),
            "errors": errors[:50],
            "model_used": model_name,
            "encoder_used": encoder_name,
        }

    def train_from_config(self, config_path: str) -> dict[str, Any]:
        from utils.config import validate_config

        config = load_config(config_path)
        cfg_errors = validate_config(config)
        if cfg_errors:
            raise ValueError("config validation failed:\n  - " + "\n  - ".join(cfg_errors))

        model_name = str(config.get("model", "deepfocus")).lower()
        encoder_name = str(config.get("encoder", "r9")).lower()
        if model_name not in self.models:
            raise ValueError(f"unknown model: {model_name}")
        if encoder_name not in self.encoders:
            raise ValueError(f"unknown encoder: {encoder_name}")
        expected = self.expected_pairs.get(model_name)
        if expected and encoder_name != expected:
            raise ValueError(f"model {model_name} expects encoder {expected}, got {encoder_name}")

        model = self.models[model_name]
        dataset_files = resolve_dataset_files(config)
        if not dataset_files:
            raise ValueError("config must define dataset_files")

        sampling = config.get("sampling", {})
        positive_cap = sampling.get("positive_cap", 5000)
        negative_cap = sampling.get("negative_cap", 15000)
        seed = int(config.get("seed", 42))
        records = []
        for dataset_file in dataset_files:
            dataset_name = Path(dataset_file).stem
            records.extend(
                collect_balanced_records(
                    dataset_file,
                    dataset=dataset_name,
                    positive_cap=positive_cap,
                    negative_cap=negative_cap,
                    seed=seed,
                )
            )
        train_records, val_records, test_records = train_test_split_71515(records, seed=seed)
        training = config.get("training", {})
        epochs = int(training.get("epochs", 8))
        learning_rate = float(training.get("learning_rate", 0.05))
        weight_decay = float(training.get("weight_decay", 1e-4))
        patience = int(training.get("patience", 5))
        pos_weight = training.get("pos_weight")
        if pos_weight is not None:
            pos_weight = float(pos_weight)

        trained_model, history = fit_linear_model(
            model,
            train_records,
            val_records,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            seed=seed,
            pos_weight=pos_weight,
        )
        metrics = evaluate_model(trained_model, test_records)
        weights_path = config.get("weights_path")
        if weights_path:
            trained_model.save(weights_path)
            self.loaded_from[model_name] = str(weights_path)
        self.models[model_name] = trained_model
        return {
            "model": model_name,
            "encoder": encoder_name,
            "train_size": len(train_records),
            "val_size": len(val_records),
            "test_size": len(test_records),
            "test_metrics": metrics,
            "history": serialize_history(history),
            "weights_path": weights_path,
        }

    def submit_train_job(self, config_path: str, temp_config_path: bool = False) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        self.jobs[job_id] = {
            "status": "queued",
            "config_path": config_path,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
        }

        def _worker():
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["started_at"] = time.time()
            try:
                result = self.train_from_config(config_path)
                self.jobs[job_id]["result"] = result
                self.jobs[job_id]["status"] = "succeeded"
            except Exception as exc:
                self.jobs[job_id]["error"] = str(exc)
                self.jobs[job_id]["status"] = "failed"
                print(f"[server] job {job_id} failed: {exc}")
            finally:
                self.jobs[job_id]["finished_at"] = time.time()
                if temp_config_path:
                    try:
                        Path(config_path).unlink(missing_ok=True)
                    except Exception:
                        pass

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return {"status": "queued", "job_id": job_id, "config_path": config_path}

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(f"unknown job_id: {job_id}")
        result = {
            "job_id": job_id,
            "status": job["status"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "error": job["error"],
        }
        if job["status"] == "succeeded" and job["result"] is not None:
            result["result"] = {
                "model": job["result"].get("model"),
                "encoder": job["result"].get("encoder"),
                "train_size": job["result"].get("train_size"),
                "test_metrics": job["result"].get("test_metrics"),
                "weights_path": job["result"].get("weights_path"),
            }
        return result


def _create_handler(registry: ModelRegistry):
    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                _json_response(self, HTTPStatus.OK, registry.health())
                return
            if self.path == "/models":
                _json_response(self, HTTPStatus.OK, registry.list_models())
                return
            if self.path.startswith("/jobs/"):
                job_id = self.path[len("/jobs/"):]
                try:
                    result = registry.get_job(job_id)
                    _json_response(self, HTTPStatus.OK, result)
                except ValueError as exc:
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            if self.path in {"/", "/index.html"}:
                index_path = Path("static/index.html")
                if index_path.exists():
                    _plain_response(self, HTTPStatus.OK, index_path.read_bytes(), content_type="text/html; charset=utf-8")
                else:
                    _plain_response(
                        self,
                        HTTPStatus.OK,
                        b"CRISPR-DualPred server is running.\n",
                    )
                return
            safe_path = _safe_static_path(self.path)
            if safe_path is not None:
                content_type = "text/plain; charset=utf-8"
                if safe_path.suffix == ".html":
                    content_type = "text/html; charset=utf-8"
                elif safe_path.suffix == ".js":
                    content_type = "application/javascript; charset=utf-8"
                elif safe_path.suffix == ".css":
                    content_type = "text/css; charset=utf-8"
                _plain_response(self, HTTPStatus.OK, safe_path.read_bytes(), content_type=content_type)
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                return

            try:
                if self.path == "/predict":
                    request = PredictRequest.model_validate(payload)
                    result = registry.predict(request.model, request.encoder, request.sgRNA, request.dna)
                    _json_response(self, HTTPStatus.OK, result)
                    return
                if self.path == "/predict/batch":
                    request = PredictBatchRequest.model_validate(payload)
                    result = registry.predict_batch(request.model, request.encoder, request.pairs)
                    _json_response(self, HTTPStatus.OK, {"results": result})
                    return
                if self.path == "/predict/file":
                    request = PredictFileRequest.model_validate(payload)
                    start = time.time()
                    result = registry.predict_from_file(
                        request.model, request.encoder, request.file_content, request.format, request.has_header
                    )
                    result["time_ms"] = int((time.time() - start) * 1000)
                    _json_response(self, HTTPStatus.OK, result)
                    return
                if self.path == "/train":
                    request = TrainRequest.model_validate(payload)
                    config_path = request.config_path or request.config
                    temp_config_path = False
                    if not config_path:
                        if request.dataset_files is None or request.model is None or request.encoder is None:
                            raise ValueError("train requires config_path or model, encoder, dataset_files")
                        temp_config = {
                            "model": request.model,
                            "encoder": request.encoder,
                            "dataset_files": request.dataset_files,
                            "weights_path": f"artifacts/{request.model}_{request.encoder}.json",
                        }
                        config_path = f"/tmp/{uuid.uuid4().hex}.json"
                        Path(config_path).write_text(json.dumps(temp_config, indent=2), encoding="utf-8")
                        temp_config_path = True
                    if request.async_run:
                        job = registry.submit_train_job(config_path, temp_config_path=temp_config_path)
                        _json_response(self, HTTPStatus.ACCEPTED, job)
                    else:
                        try:
                            result = registry.train_from_config(config_path)
                            _json_response(self, HTTPStatus.OK, result)
                        finally:
                            if temp_config_path:
                                try:
                                    Path(config_path).unlink(missing_ok=True)
                                except Exception:
                                    pass
                    return
            except ValidationError as exc:
                _json_response(self, HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "validation failed", "details": exc.errors()})
                return
            except Exception as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


def build_fastapi_app(registry: ModelRegistry):  # pragma: no cover - optional dependency support
    if not FASTAPI_AVAILABLE:
        return None

    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="CRISPR-DualPred")

    @app.exception_handler(ValueError)
    def _value_error_handler(request, exc):
        msg = str(exc).lower()
        status = 404 if "unknown job_id" in msg else 400
        return JSONResponse(status_code=status, content={"error": str(exc)})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/index.html", response_class=HTMLResponse)
    def index():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return PlainTextResponse("CRISPR-DualPred server is running.\n")

    @app.get("/health")
    def health():
        return registry.health()

    @app.post("/predict")
    def predict(request: PredictRequest):
        return registry.predict(request.model, request.encoder, request.sgRNA, request.dna)

    @app.post("/predict/batch")
    def predict_batch(request: PredictBatchRequest):
        return {"results": registry.predict_batch(request.model, request.encoder, request.pairs)}

    @app.get("/models")
    def models():
        return registry.list_models()

    @app.post("/predict/file")
    def predict_file(request: PredictFileRequest):
        start = time.time()
        result = registry.predict_from_file(
            request.model, request.encoder, request.file_content, request.format, request.has_header
        )
        result["time_ms"] = int((time.time() - start) * 1000)
        return result

    @app.get("/jobs/{job_id}")
    def get_job_status(job_id: str):
        return registry.get_job(job_id)

    @app.post("/train")
    def train(request: TrainRequest):
        config_path = request.config_path or request.config
        temp_config_path = False
        if not config_path:
            if request.dataset_files is None or request.model is None or request.encoder is None:
                raise ValueError("train requires config_path or model, encoder, dataset_files")
            temp_config = {
                "model": request.model,
                "encoder": request.encoder,
                "dataset_files": request.dataset_files,
                "weights_path": f"artifacts/{request.model}_{request.encoder}.json",
            }
            config_path = f"/tmp/{uuid.uuid4().hex}.json"
            Path(config_path).write_text(json.dumps(temp_config, indent=2), encoding="utf-8")
            temp_config_path = True

        if request.async_run:
            return registry.submit_train_job(config_path, temp_config_path=temp_config_path)

        try:
            result = registry.train_from_config(config_path)
            return result
        finally:
            if temp_config_path:
                try:
                    Path(config_path).unlink(missing_ok=True)
                except Exception:
                    pass

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CRISPR-DualPred local API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/r9_deepfocus.yaml", "configs/c9_conmismatch9.yaml"],
        help="Configuration files used to bootstrap model weights",
    )
    args = parser.parse_args()

    registry = ModelRegistry(config_paths=args.config, device=args.device)

    if FASTAPI_AVAILABLE:
        try:
            import uvicorn  # type: ignore

            app = build_fastapi_app(registry)
            if app is not None:
                uvicorn.run(app, host=args.host, port=args.port, log_level="info")
                return
        except Exception:
            pass

    handler = _create_handler(registry)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"CRISPR-DualPred server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
