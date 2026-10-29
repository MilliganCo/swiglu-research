from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ops import (hosvd_bases, mlp, permutation, permute_weights, relative_error,
                  signatures, tensorize, tucker, untensorize, validate_shape)
from .telemetry import PowerSampler, append_jsonl, environment, snapshot_source, source_hash, write_json


def configure(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)


def validate_config(config: dict[str, Any]) -> None:
    if config["device"] != "cuda" or config["dtype"] != "bfloat16":
        raise ValueError("phase 1 requires CUDA and unquantized BF16")
    if config["layer"] != 14:
        raise ValueError("first deliverable operates only on zero-based layer 14")
    validate_shape(tuple(config["shape"]), 6144, 2048)
    ranks = config["ranks"]
    if not ranks or any(len(r) != 5 or any(v < 1 or v > d for v, d in zip(r, config["shape"])) for r in ranks):
        raise ValueError("invalid Tucker ranks")
    if set(config["strategies"]) != {"identity", "random", "structure-aware"}:
        raise ValueError("all three permutation strategies are required")
    quality = config["quality"]
    if not 1 <= quality["stride"] <= quality["block_size"]:
        raise ValueError("quality stride must fit the context window")
    if min(quality["evaluation_tokens"], quality["calibration_tokens"], quality["logits_tokens"]) < 2:
        raise ValueError("quality requires at least two tokens")
    bench = config["benchmark"]
    if bench["batch_size"] != 1 or min(bench["iterations"], bench["warmups"], bench["new_tokens"]) < 1:
        raise ValueError("invalid benchmark settings")


def model_weights(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: getattr(module, name + "_proj").weight.detach().cpu().clone()
            for name in ("gate", "up", "down")}


@torch.no_grad()
def install_weights(module: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
    for name, weight in weights.items():
        getattr(module, name + "_proj").weight.copy_(weight)


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


class Experiment:
    def __init__(self, config: dict[str, Any], root: Path, smoke: bool) -> None:
        self.config, self.root, self.smoke = config, root, smoke
        self.output = root / config["output"]
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        self.directory = self.output / "runs" / self.run_id
        self.directory.mkdir(parents=True)
        (self.directory / "factors").mkdir()
        self.rows: list[dict[str, Any]] = []
        self.metadata = environment(root)
        self.metadata["source_snapshot_sha256"] = snapshot_source(root, self.directory)
        if not self.metadata["wsl"] or not torch.cuda.is_available():
            raise RuntimeError("run development and benchmarks in WSL2 with CUDA")
        self.metadata.update({"run_id": self.run_id, "smoke": smoke, "config": config,
                              "dtype": config["dtype"], "seed": config["seed"],
                              "hf_model_revision_sha": None, "dataset_revision_sha": None})
        write_json(self.directory / "metadata.json", self.metadata)

    def record(self, kind: str, **values: Any) -> dict[str, Any]:
        row = {**self.metadata, "timestamp": datetime.now(timezone.utc).isoformat(),
               "kind": kind, "status": "ok", **values}
        append_jsonl(self.output / "experiments.jsonl", row)
        append_jsonl(self.directory / "raw.jsonl", row)
        self.rows.append(row)
        print(json.dumps({k: row[k] for k in ("kind", "status", "strategy", "seed", "ranks", "prompt_length", "mlp_output_relative_error") if k in row}), flush=True)
        return row

    def load(self) -> None:
        config = self.config
        api = HfApi()
        revision = api.model_info(config["model"], revision=config["revision"]).sha
        self.metadata["hf_model_revision_sha"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"], revision=revision, dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=False).eval().to("cuda")
        architecture = self.model.config
        if (architecture.hidden_size, architecture.intermediate_size, architecture.num_hidden_layers) != (2048, 6144, 28):
            raise ValueError("loaded model does not match the requested architecture")
        self.metadata["model_parameter_count"] = sum(p.numel() for p in self.model.parameters())
        self.metadata["model_weight_bytes"] = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.metadata["attention_implementation"] = architecture._attn_implementation
        self.module = self.model.model.layers[config["layer"]].mlp
        self.weights = model_weights(self.module)
        save_file(self.weights, str(self.directory / "layer14.safetensors"),
                  metadata={"model": config["model"], "revision": revision, "layer": "14"})
        quality = config["quality"]
        dataset_revision = api.dataset_info(quality["dataset"], revision=quality["revision"]).sha
        self.metadata["dataset_revision_sha"] = dataset_revision
        tokens = {}
        for split, count in (("train", quality["calibration_tokens"]), ("test", quality["evaluation_tokens"])):
            dataset = load_dataset(quality["dataset"], quality["subset"], split=split, revision=dataset_revision)
            text = "\n\n".join(dataset["text"])
            ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, verbose=False)["input_ids"][:, :count]
            if ids.shape[1] != count:
                raise ValueError(f"not enough {split} tokens")
            tokens[split] = ids.to("cuda")
            np.save(self.directory / f"{split}_tokens.npy", ids.numpy())
            self.metadata[f"{split}_token_sha256"] = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        self.train, self.test = tokens["train"], tokens["test"]
        self.metadata["quality_scope"] = {"dataset": quality["dataset"], "subset": quality["subset"],
                                          "split": "test", "tokens": quality["evaluation_tokens"],
                                          "full_evaluation": False}
        write_json(self.directory / "metadata.json", self.metadata)
        self.record("environment", prompt_length=None, batch_size=1, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                    latency_ms=None, tokens_per_second=None, quality_metrics=None)

    @torch.inference_mode()
    def perplexity(self) -> dict[str, Any]:
        quality = self.config["quality"]
        total_loss, count, previous_end = 0.0, 0, 0
        for start in range(0, self.test.shape[1], quality["stride"]):
            end = min(start + quality["block_size"], self.test.shape[1])
            ids = self.test[:, start:end]
            logits = self.model(ids, use_cache=False).logits
            begin_target = max(previous_end - start - 1, 0)
            # Float32 CE in chunks avoids a second full-vocabulary activation buffer.
            for offset in range(begin_target, ids.shape[1] - 1, 64):
                stop = min(offset + 64, ids.shape[1] - 1)
                loss = F.cross_entropy(logits[:, offset:stop].float().reshape(-1, logits.shape[-1]),
                                       ids[:, offset + 1:stop + 1].reshape(-1), reduction="sum")
                total_loss += loss.item()
                count += stop - offset
            del logits
            previous_end = end
            if end == self.test.shape[1]:
                break
        if count != self.test.shape[1] - 1:
            raise AssertionError("every next-token target must be scored exactly once")
        return {"wikitext2_raw_perplexity": math.exp(total_loss / count), "nll": total_loss / count,
                "scored_tokens": count, "full_evaluation": False,
                "block_size": quality["block_size"], "stride": quality["stride"]}

    @torch.inference_mode()
    def capture(self) -> None:
        activations = []

        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            activations.append(inputs[0].detach().cpu())

        handle = self.module.register_forward_pre_hook(hook)
        try:
            for block in self.train.split(self.config["quality"]["block_size"], dim=1):
                self.model(block, use_cache=False, logits_to_keep=1)
        finally:
            handle.remove()
        self.x = torch.cat(activations, dim=1).to("cuda")
        self.original_output = self.module(self.x).float()
        self.logit_ids = self.test[:, :self.config["quality"]["logits_tokens"]]
        self.original_logits = self.model(self.logit_ids, use_cache=False).logits.float()
        save_file({"mlp_inputs": self.x.cpu()}, str(self.directory / "calibration.safetensors"))

    @torch.inference_mode()
    def logits_metrics(self) -> dict[str, float]:
        actual = self.model(self.logit_ids, use_cache=False).logits.float()
        original = self.original_logits
        logp, logq = original.log_softmax(dim=-1), actual.log_softmax(dim=-1)
        return {"final_logits_cosine_similarity": F.cosine_similarity(actual.flatten(), original.flatten(), dim=0).item(),
                "final_logits_relative_error": relative_error(actual, original),
                "final_logits_max_error": (actual - original).abs().max().item(),
                "kl_dense_to_candidate": (logp.exp() * (logp - logq)).sum(dim=-1).mean().item()}

    @torch.inference_mode()
    def symmetry(self, strategy: str, seed: int, perm: torch.Tensor) -> None:
        tolerance = self.config["tolerance"]
        metrics = {}
        for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            w = {key: value.to(device="cuda", dtype=dtype) for key, value in self.weights.items()}
            x = self.x.to(dtype)
            original = mlp(x, w)
            actual = mlp(x, permute_weights(w, perm.to("cuda")))
            torch.testing.assert_close(actual, original, rtol=tolerance[name + "_rtol"], atol=tolerance[name + "_atol"])
            relative = relative_error(actual, original)
            if name == "bf16" and relative > tolerance["bf16_relative"]:
                raise AssertionError(f"BF16 relative symmetry error {relative}")
            metrics[name] = {"relative_error": relative, "max_error": (actual - original).abs().max().item(),
                             "rtol": tolerance[name + "_rtol"], "atol": tolerance[name + "_atol"]}
            del w, x, original, actual
        try:
            install_weights(self.module, permute_weights(self.weights, perm))
            logits = self.logits_metrics()
            if (logits["final_logits_relative_error"] > tolerance["logits_relative"] or
                    logits["final_logits_max_error"] > tolerance["logits_max"]):
                raise AssertionError(f"permutation changed model logits beyond tolerance: {logits}")
        finally:
            install_weights(self.module, self.weights)
        self.record("symmetry", strategy=strategy, seed=seed, mlp_metrics=metrics, quality_metrics=logits,
                    tolerance=tolerance, permutation_sha256=hashlib.sha256(perm.numpy().tobytes()).hexdigest())

    @torch.inference_mode()
    def benchmark(self, quality: dict[str, Any]) -> None:
        bench = self.config["benchmark"]
        for length in bench["prompt_lengths"]:
            try:
                # Stable natural-text prefix; repeated only if the requested prompt exceeds the subset.
                prompt = self.test.repeat(1, math.ceil(length / self.test.shape[1]))[:, :length]

                def generate() -> torch.Tensor:
                    return self.model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                               max_new_tokens=bench["new_tokens"], min_new_tokens=bench["new_tokens"],
                                               do_sample=False, use_cache=True,
                                               pad_token_id=self.tokenizer.eos_token_id)

                for _ in range(bench["warmups"]):
                    generate()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                elapsed, wall, token_counts = [], [], []
                sampler = PowerSampler()
                sampler.start()
                try:
                    for index in range(bench["iterations"]):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        wall_start = time.perf_counter()
                        start.record()
                        output = generate()
                        end.record()
                        torch.cuda.synchronize()
                        elapsed.append(start.elapsed_time(end))
                        wall.append((time.perf_counter() - wall_start) * 1000)
                        token_counts.append(output.shape[1] - length)
                        if token_counts[-1] != bench["new_tokens"]:
                            raise AssertionError("generation did not produce the fixed token count")
                        del output
                        if index % 10 == 9:
                            print(f"baseline prompt={length} measured={index + 1}/{bench['iterations']}", flush=True)
                finally:
                    power = sampler.finish(sum(token_counts))
                self.record("baseline", prompt_length=length, batch_size=bench["batch_size"],
                            new_tokens=bench["new_tokens"], warmups=bench["warmups"], iterations=bench["iterations"],
                            cuda_event_samples_ms=elapsed, wall_samples_ms=wall, generated_token_counts=token_counts,
                            latency_ms=statistics.mean(elapsed), latency_p50_ms=float(np.median(elapsed)),
                            latency_p95_ms=float(np.quantile(elapsed, 0.95)),
                            tokens_per_second=bench["new_tokens"] * 1000 / statistics.mean(elapsed),
                            wall_tokens_per_second=bench["new_tokens"] * 1000 / statistics.mean(wall),
                            peak_vram_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
                            quality_metrics=quality, **power)
                cleanup()
            except torch.cuda.OutOfMemoryError as error:
                self.record("baseline", status="oom", prompt_length=length, error=str(error))
                cleanup()

    def reuse_baseline(self, directory: Path) -> None:
        previous = json.loads((directory / "metadata.json").read_text())
        keys = ("hf_model_revision_sha", "dataset_revision_sha", "dtype", "torch_version",
                "transformers_version", "cuda_runtime", "nvidia_driver", "gpu", "test_token_sha256")
        if any(previous[key] != self.metadata[key] for key in keys):
            raise ValueError("baseline model/data/runtime do not match the current run")
        if previous["smoke"] or previous["config"]["benchmark"] != self.config["benchmark"]:
            raise ValueError("baseline measurement settings do not match")
        rows = [json.loads(line) for line in (directory / "raw.jsonl").read_text().splitlines()]
        baseline = [r for r in rows if r["kind"] == "baseline"]
        if (len(baseline) != len(self.config["benchmark"]["prompt_lengths"]) or
                {r["prompt_length"] for r in baseline} != set(self.config["benchmark"]["prompt_lengths"]) or
                any(r["status"] != "ok" for r in baseline)):
            raise ValueError("baseline is incomplete or contains OOM")
        self.metadata["reused_baseline_run_id"] = previous["run_id"]
        diagnostic = directory / "precision_diagnostic.json"
        if diagnostic.exists():
            self.metadata["precision_diagnostic"] = json.loads(diagnostic.read_text())
            self.metadata["original_logits_tolerance"] = previous["config"]["tolerance"]
        write_json(self.directory / "metadata.json", self.metadata)
        for row in baseline:
            values = {key: value for key, value in row.items()
                      if key not in previous and key not in ("kind", "status", "timestamp")}
            self.record("baseline", **values, measurement_run_id=previous["run_id"],
                        measurement_timestamp=row["timestamp"], measurement_source_sha256=previous["source_sha256"])

    @torch.inference_mode()
    def sweep(self, dense_quality: dict[str, Any]) -> None:
        shape = tuple(self.config["shape"])
        original_parameters = sum(w.numel() for w in self.weights.values())
        for strategy in self.config["strategies"]:
            seeds = self.config["permutation_seeds"][:1] if strategy == "identity" else self.config["permutation_seeds"]
            for seed in seeds:
                perm = permutation(strategy, self.weights, seed, self.config["clusters"])
                save_file({"permutation": perm, "signatures": signatures(self.weights, seed)},
                          str(self.directory / f"permutation-{strategy}-{seed}.safetensors"))
                self.symmetry(strategy, seed, perm)
                originals, bases = {}, {}
                try:
                    for name, weight in permute_weights(self.weights, perm).items():
                        originals[name] = tensorize(weight.to(device="cuda", dtype=torch.float32), shape, down=name == "down")
                        bases[name] = hosvd_bases(originals[name])
                except torch.cuda.OutOfMemoryError as error:
                    for ranks in self.config["ranks"]:
                        self.record("rank_sweep", status="oom", strategy=strategy, seed=seed, ranks=ranks, error=str(error))
                    originals.clear()
                    bases.clear()
                    cleanup()
                    continue
                for rank_list in self.config["ranks"]:
                    ranks = tuple(rank_list)
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    reconstructed, packed, weight_metrics = {}, {"permutation": perm}, {}
                    try:
                        parameter_count, squared_error, squared_norm = 0, 0.0, 0.0
                        for name, tensor in originals.items():
                            decomposition = tucker(tensor, ranks, bases[name])
                            recovered = decomposition.reconstruct()
                            error = recovered - tensor
                            weight_metrics[name] = {"relative_frobenius_error": relative_error(recovered, tensor),
                                                    "max_error": error.abs().max().item(),
                                                    "original_parameters": tensor.numel(),
                                                    "factorized_parameters": decomposition.parameters}
                            squared_error += error.square().sum().item()
                            squared_norm += tensor.square().sum().item()
                            parameter_count += decomposition.parameters
                            reconstructed[name] = untensorize(recovered, down=name == "down").to(torch.bfloat16)
                            packed[name + ".core"] = decomposition.core.cpu().to(torch.bfloat16)
                            for axis, factor in enumerate(decomposition.factors):
                                packed[f"{name}.factor{axis}"] = factor.cpu().to(torch.bfloat16)
                            del decomposition, recovered, error
                        # The saved BF16 factors, rather than an optimistic FP32-only approximation,
                        # define the candidate used for quality metrics.
                        from .ops import Tucker
                        for name in originals:
                            saved = Tucker(packed[name + ".core"].to("cuda").float(),
                                           [packed[f"{name}.factor{axis}"].to("cuda").float() for axis in range(5)])
                            reconstructed[name] = untensorize(saved.reconstruct(), down=name == "down").to(torch.bfloat16)
                        del saved
                        output = mlp(self.x, reconstructed).float()
                        functional_error = relative_error(output, self.original_output)
                        mse = (output - self.original_output).square().mean().item()
                        install_weights(self.module, reconstructed)
                        quality = self.perplexity()
                        quality.update(self.logits_metrics())
                        quality["perplexity_change_percent"] = 100 * (quality["wikitext2_raw_perplexity"] / dense_quality["wikitext2_raw_perplexity"] - 1)
                        save_file(packed, str(self.directory / "factors" / f"{strategy}-{seed}-{'x'.join(map(str, ranks))}.safetensors"))
                        self.record("rank_sweep", strategy=strategy, seed=seed, shape=shape, ranks=ranks,
                                    original_parameters=original_parameters, factorized_parameters=parameter_count,
                                    compression_ratio=original_parameters / parameter_count,
                                    dense_weight_bytes=original_parameters * 2, factorized_weight_bytes=parameter_count * 2,
                                    permutation_storage_bytes=perm.numel() * perm.element_size(),
                                    relative_frobenius_error=math.sqrt(squared_error / squared_norm),
                                    max_error=max(v["max_error"] for v in weight_metrics.values()), weight_metrics=weight_metrics,
                                    mlp_output_relative_error=functional_error, block_output_mse=mse,
                                    quality_metrics=quality, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                                    elapsed_seconds=time.perf_counter() - started,
                                    quality_weight_representation="reconstructed BF16 core/factors; dense matmul; no speedup claim")
                    except torch.cuda.OutOfMemoryError as error:
                        self.record("rank_sweep", status="oom", strategy=strategy, seed=seed, shape=shape, ranks=ranks, error=str(error))
                    finally:
                        install_weights(self.module, self.weights)
                        reconstructed.clear()
                        packed.clear()
                        saved = decomposition = recovered = output = None
                        cleanup()
                originals.clear()
                bases.clear()
                cleanup()

    def report(self) -> None:
        analysis_sha256 = source_hash(self.root)
        rows = [r for r in self.rows if r["kind"] == "rank_sweep" and r["status"] == "ok"]
        baseline = [r for r in self.rows if r["kind"] == "baseline"]
        lines = ["# SwiGLU-Fold — отчёт первого deliverable", "",
                 f"Run: `{self.run_id}`. Smoke: `{self.smoke}`.", "",
                 "Исследован только слой 14 (нумерация с нуля), без quantization и без обучения.",
                 f"Модель: `{self.config['model']}`, revision `{self.metadata['hf_model_revision_sha']}`.",
                 f"GPU: {self.metadata['gpu']}, CUDA {self.metadata['cuda_runtime']}, driver {self.metadata['nvidia_driver']}.",
                 f"PyTorch {self.metadata['torch_version']}, Transformers {self.metadata['transformers_version']}.",
                 f"Git commit: {self.metadata['git_commit']}; source SHA256: `{self.metadata['source_sha256']}`.", "",
                 f"Analysis source SHA256: `{analysis_sha256}` (отдельно от исходников измеренного run).", "",
                 "## Метод", "",
                 "Gate/up rows и down columns переставлены совместно. Signatures: отдельно нормализованные",
                 "gate/up/down, три Gaussian projections по 32 координаты, MiniBatchKMeans и PC1 внутри cluster.",
                 "Для каждой матрицы вычислены собственные HOSVD bases через eigh Gram matrices; ranks усечены",
                 "одинаково для всех стратегий. Down tensorization использует транспонированную логическую матрицу.",
                 "Weights/decomposition errors относятся к FP32 HOSVD; functional/PPL/logits — к сохранённым BF16",
                 "factors/core после reconstruction. Скорость factorized inference здесь не измерялась.", "",
                 "Calibration: начало train WikiText-2 raw; quality: отдельное начало test, все next-token targets",
                 "учтены ровно один раз в sliding windows. Это ограниченный subset, не полный benchmark.", "",
                 "## Dense baseline", "", "| Prompt | Iterations | Latency ms | Tokens/s | Peak allocated MiB | Status |",
                 "|---:|---:|---:|---:|---:|---|"]
        for row in baseline:
            if row["status"] == "ok":
                lines.append(f"| {row['prompt_length']} | {row['iterations']} | {row['latency_ms']:.2f} | {row['tokens_per_second']:.2f} | {row['peak_vram_bytes']/2**20:.1f} | ok |")
            else:
                lines.append(f"| {row['prompt_length']} | — | — | — | — | {row['status']} |")
        dense = next((r for r in self.rows if r["kind"] == "dense_quality"), None)
        if dense:
            quality = dense["quality_metrics"]
            lines += ["", f"Dense subset perplexity: **{quality['wikitext2_raw_perplexity']:.4f}**, scored tokens: {quality['scored_tokens']}."]
        if "reused_baseline_run_id" in self.metadata:
            lines += ["", f"Baseline переиспользован из run `{self.metadata['reused_baseline_run_id']}` после проверки совпадения model/data/runtime и measurement settings. Четыре серии по 30 samples измерены там; повторная запись содержит provenance."]
        if "precision_diagnostic" in self.metadata:
            previous = self.metadata["original_logits_tolerance"]
            current = self.config["tolerance"]
            lines += ["", "### Численная точность", "",
                      f"Исходный допуск BF16 logits {previous['logits_relative']*100:g}% / max {previous['logits_max']:g} оказался ниже накопленной погрешности. Неудачный run сохранён.",
                      "Отдельная проверка полной модели (FP32 на CPU, BF16 на GPU):"]
            for check in self.metadata["precision_diagnostic"]["rows"]:
                label = check.get("comparison", f"FP32 {check.get('strategy', '')}")
                lines.append(f"- {label}: relative={check['relative_error']:.9g}, max={check['max_error']:.9g}.")
            lines += [f"До повторного sweep установлен BF16 logits допуск {current['logits_relative']*100:g}% / max {current['logits_max']:g}.",
                      "Это отдельная численная проверка; quality guardrail PPL ≤5% не изменялся."]
        lines += ["", "## Exact permutation checks", "",
                  "| Strategy | Seed | FP32 MLP rel. | BF16 MLP rel. | BF16 MLP max | Logits rel. | Logits max |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        symmetry_rows = [r for r in self.rows if r["kind"] == "symmetry" and r["status"] == "ok"]
        for row in symmetry_rows:
            m, q = row["mlp_metrics"], row["quality_metrics"]
            lines.append(f"| {row['strategy']} | {row['seed']} | {m['fp32']['relative_error']:.6g} | {m['bf16']['relative_error']:.6g} | {m['bf16']['max_error']:.6g} | {q['final_logits_relative_error']:.6g} | {q['final_logits_max_error']:.6g} |")
        lines += ["", "## Compression ratio vs reconstruction / functional error", "",
                  "| Strategy | Seed | Ranks | Ratio | Frobenius rel. | MLP rel. | Block MSE | PPL | PPL change % | Logits cosine | KL |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            q = row["quality_metrics"]
            lines.append(f"| {row['strategy']} | {row['seed']} | {'×'.join(map(str,row['ranks']))} | {row['compression_ratio']:.3f} | {row['relative_frobenius_error']:.6f} | {row['mlp_output_relative_error']:.6f} | {row['block_output_mse']:.6g} | {q['wikitext2_raw_perplexity']:.4f} | {q['perplexity_change_percent']:+.3f} | {q['final_logits_cosine_similarity']:.6f} | {q['kl_dense_to_candidate']:.6g} |")
        lines += ["", "## Проверка основной гипотезы", ""]
        comparisons = []
        for ranks in self.config["ranks"]:
            matched = [r for r in rows if list(r["ranks"]) == ranks]
            identity = [r for r in matched if r["strategy"] == "identity"]
            structured = [r for r in matched if r["strategy"] == "structure-aware"]
            random = [r for r in matched if r["strategy"] == "random"]
            if not identity or not structured or not random:
                continue
            i = identity[0]["mlp_output_relative_error"]
            s = statistics.mean(r["mlp_output_relative_error"] for r in structured)
            r = statistics.mean(r["mlp_output_relative_error"] for r in random)
            comparisons.append(s < min(i, r))
            s_std = statistics.stdev(row["mlp_output_relative_error"] for row in structured) if len(structured) > 1 else 0.0
            r_std = statistics.stdev(row["mlp_output_relative_error"] for row in random) if len(random) > 1 else 0.0
            wins = sum(row["mlp_output_relative_error"] < control["mlp_output_relative_error"]
                       for row in structured for control in random if row["seed"] == control["seed"])
            ppl_i = identity[0]["quality_metrics"]["wikitext2_raw_perplexity"]
            ppl_s = statistics.mean(row["quality_metrics"]["wikitext2_raw_perplexity"] for row in structured)
            ppl_r = statistics.mean(row["quality_metrics"]["wikitext2_raw_perplexity"] for row in random)
            lines.append(f"- Ranks {ranks}: MLP rel. identity={i:.6f}, random={r:.6f} ± {r_std:.6f}, structure-aware={s:.6f} ± {s_std:.6f} (sample SD); улучшение C относительно identity={(i-s)/i*100:.6f}%, относительно random={(r-s)/r*100:.6f}%; C лучше random в {wins}/{len(structured)} paired seeds. PPL means: identity={ppl_i:.6f}, random={ppl_r:.6f}, C={ppl_s:.6f}.")
        if self.smoke:
            verdict = "Smoke проверяет исполнение; исследовательский вывод по smoke не делается."
        elif len(comparisons) != len(self.config["ranks"]):
            verdict = "Данных недостаточно: не все matched-budget конфигурации завершены."
        elif all(comparisons):
            verdict = "Обнаружен предварительный сигнал в пользу гипотезы по средним functional errors. Устойчивое превосходство пока не доказано: три permutation seeds и один subset недостаточны для однозначного положительного вывода, а малые differences требуют проверки численной точности. Это не подтверждает перенос на другие слои или полную модель."
        elif not any(comparisons):
            verdict = "Гипотеза не подтверждена для данной tensorization и structure-aware v1: ни один проверенный budget не улучшил среднюю functional error относительно обеих контрольных стратегий. Это отрицательный результат данного эксперимента, не доказательство невозможности иных порядков."
        else:
            verdict = "Результат смешанный: преимущество structure-aware зависит от rank budget. Общий вывод о превосходстве v1 не подтверждён."
        lines += ["", "**" + verdict + "**", "", "## Границы доказательства", "",
                  "- Не выполнены full WikiText-2, HellaSwag, PIQA, ARC Easy, WinoGrande, sensitivity и full-model compression.",
                  "- Compression ratio здесь относится только к трём матрицам MLP слоя 14; это не снижение общего footprint модели на 25%.",
                  "- Не измерены direct factorized inference и Triton; reconstructed candidates не обещают speedup.",
                  "- Seed replicas оценивают вариацию permutation; отсутствуют доверительные интервалы по корпусам и слоям.",
                  "- PyTorch peak allocated/reserved отражает процесс; Windows desktop использует общую GPU. Power может быть unavailable.",
                  "- WSL RAM ограничена текущей конфигурацией; абсолютный Git SHA отсутствует, пока в проекте нет commit.",
                  "", "## Raw results и воспроизведение", "",
                  f"Run data: `results/runs/{self.run_id}/`; общая append-only история: `results/experiments.jsonl`.",
                  "Pinned YAML и environment metadata сохранены внутри run. Команда из WSL:", "",
                  f"```bash\nswiglu-fold --config results/runs/{self.run_id}/resolved.yaml --stage all{' --smoke' if self.smoke else ''}\n```", ""]
        if symmetry_rows:
            floor = max(row["mlp_metrics"]["bf16"]["relative_error"] for row in symmetry_rows)
            lines += [f"Наблюдаемая BF16 MLP numerical error от одной exact permutation: до {floor*100:.4f}%. Это ориентир численного масштаба, не доверительный интервал.",
                      "Triton и full-model этап не начаты: сначала требуется подтвердить устойчивый algorithmic benefit.", ""]
        content = "\n".join(lines)
        (self.directory / "REPORT_PHASE1.md").write_text(content, encoding="utf-8")
        if not self.smoke:
            (self.root / "REPORT_PHASE1.md").write_text(content, encoding="utf-8")
        resolved = copy.deepcopy(self.config)
        resolved["revision"] = self.metadata["hf_model_revision_sha"]
        resolved["quality"]["revision"] = self.metadata["dataset_revision_sha"]
        (self.directory / "resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        write_json(self.directory / "summary.json", {"verdict": verdict, "completed_configurations": len(rows),
                                                   "oom_configurations": sum(r["status"] == "oom" for r in self.rows),
                                                   "analysis_source_sha256": analysis_sha256})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    parser.add_argument("--stage", choices=("env", "baseline", "sweep", "all", "report"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline-run", type=Path, help="Reuse compatible completed dense measurements, with provenance")
    parser.add_argument("--report-run", type=Path, help="Regenerate analysis from existing raw measurements")
    args = parser.parse_args()
    root = Path.cwd()
    if args.stage == "report":
        if not args.report_run:
            parser.error("--stage report requires --report-run")
        run = Experiment.__new__(Experiment)
        run.directory, run.root = args.report_run, root
        run.metadata = json.loads((run.directory / "metadata.json").read_text())
        run.config, run.run_id, run.smoke = run.metadata["config"], run.metadata["run_id"], run.metadata["smoke"]
        run.rows = [json.loads(line) for line in (run.directory / "raw.jsonl").read_text().splitlines()]
        run.report()
        return
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.smoke:
        config["benchmark"].update(prompt_lengths=[128], warmups=1, iterations=1, new_tokens=8)
        config["quality"].update(calibration_tokens=64, evaluation_tokens=512, logits_tokens=16)
        config["permutation_seeds"] = [config["seed"]]
        config["ranks"] = [config["ranks"][0]]
    configure(config["seed"])
    run = Experiment(config, root, args.smoke)
    print("RUN_DIRECTORY", run.directory, flush=True)
    if args.stage == "env":
        run.record("environment")
        return
    try:
        run.load()
        dense_quality = run.perplexity()
        run.record("dense_quality", quality_metrics=dense_quality)
        if args.stage in ("baseline", "all"):
            if args.baseline_run:
                run.reuse_baseline(args.baseline_run)
            else:
                run.benchmark(dense_quality)
        if args.stage in ("sweep", "all"):
            run.capture()
            run.sweep(dense_quality)
        run.report()
    except Exception as error:
        run.record("failure", status="failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
