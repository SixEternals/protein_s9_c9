from __future__ import annotations

import argparse
import json
import os
import threading
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
    from fastapi.responses import JSONResponse

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

    state_dict = checkpoint.get("model_state_dict", checkpoint)
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

        if config_paths:
            for config_path in config_paths:
                self._load_configured_weights(config_path)

    def _load_configured_weights(self, config_path: str) -> None:
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            return
        except Exception:
            return

        model_name = str(config.get("model", "")).lower()
        weights_path = config.get("weights_path")
        if model_name not in self.models or not weights_path:
            return

        weights = Path(weights_path)
        if not weights.exists():
            return

        if weights.suffix in {".pt", ".pth"}:
            try:
                self.models[model_name] = _load_torch_predictor(model_name, weights, self.device)
                self.loaded_from[model_name] = str(weights)
            except Exception:
                return
            return

        try:
            model_cls = type(self.models[model_name])
            state = model_cls.load_state(weights)
            self.models[model_name] = model_cls(state=state)
            self.loaded_from[model_name] = str(weights)
        except Exception:
            return

    def _model_backend(self, model_name: str) -> str:
        model = self.models[model_name]
        return str(getattr(model, "model_backend", "legacy_json"))

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "available_encoders": list(self.encoders.keys()),
            "available_models": list(self.models.keys()),
            "device": self.device,
            "loaded_models": sorted(self.loaded_from.keys()),
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

    def train_from_config(self, config_path: str) -> dict[str, Any]:
        config = load_config(config_path)
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
            if self.path.startswith("/static/"):
                relative = self.path[len("/static/") :]
                static_path = Path("static") / relative
                if static_path.exists() and static_path.is_file():
                    content_type = "text/plain; charset=utf-8"
                    if static_path.suffix == ".html":
                        content_type = "text/html; charset=utf-8"
                    elif static_path.suffix == ".js":
                        content_type = "application/javascript; charset=utf-8"
                    elif static_path.suffix == ".css":
                        content_type = "text/css; charset=utf-8"
                    _plain_response(self, HTTPStatus.OK, static_path.read_bytes(), content_type=content_type)
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
                        job_id = uuid.uuid4().hex

                        def _worker():
                            try:
                                registry.train_from_config(config_path)
                            finally:
                                if temp_config_path:
                                    try:
                                        Path(config_path).unlink(missing_ok=True)
                                    except Exception:
                                        pass

                        thread = threading.Thread(target=_worker, daemon=True)
                        thread.start()
                        _json_response(self, HTTPStatus.ACCEPTED, {"status": "queued", "job_id": job_id, "config_path": config_path})
                    else:
                        result = registry.train_from_config(config_path)
                        _json_response(self, HTTPStatus.OK, result)
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

    app = FastAPI(title="CRISPR-DualPred")

    @app.get("/health")
    def health():
        return registry.health()

    @app.post("/predict")
    def predict(request: PredictRequest):
        return registry.predict(request.model, request.encoder, request.sgRNA, request.dna)

    @app.post("/predict/batch")
    def predict_batch(request: PredictBatchRequest):
        return {"results": registry.predict_batch(request.model, request.encoder, request.pairs)}

    @app.post("/train")
    def train(request: TrainRequest):
        config_path = request.config_path or request.config
        if config_path:
            return registry.train_from_config(config_path)
        if request.dataset_files and request.model and request.encoder:
            temp_config = {
                "model": request.model,
                "encoder": request.encoder,
                "dataset_files": request.dataset_files,
                "weights_path": f"artifacts/{request.model}_{request.encoder}.json",
            }
            config_path = f"/tmp/{uuid.uuid4().hex}.json"
            Path(config_path).write_text(json.dumps(temp_config, indent=2), encoding="utf-8")
            return registry.train_from_config(config_path)
        raise ValueError("train requires config_path or model, encoder, dataset_files")

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
