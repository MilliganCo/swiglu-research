"""Assert first-deliverable acceptance against saved measurements, not log messages."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import prod
import struct
import zipfile
from pathlib import Path


def verify(directory: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in (directory / "raw.jsonl").read_text().splitlines()]
    meta = json.loads((directory / "metadata.json").read_text())
    config = meta["config"]
    assert not meta["smoke"], "smoke is not research acceptance"
    assert meta["wsl"] and meta["compute_capability"] == [8, 9]
    assert len(meta["hf_model_revision_sha"]) == 40
    assert meta["dtype"] == "bfloat16" and config["layer"] == 14
    assert all(r["status"] == "ok" for r in rows), "run contains failed or OOM configurations"
    baseline = [r for r in rows if r["kind"] == "baseline"]
    assert {r["prompt_length"] for r in baseline} == {128, 512, 2048, 4096}
    for row in baseline:
        assert row["warmups"] == 10 and row["iterations"] >= 30 and row["batch_size"] == 1
        assert len(row["cuda_event_samples_ms"]) == row["iterations"]
        assert len(row["wall_samples_ms"]) == row["iterations"]
        assert row["new_tokens"] == 128 and set(row["generated_token_counts"]) == {128}
        assert all(t > 0 for t in row["cuda_event_samples_ms"])
        assert row["peak_vram_bytes"] > 0
        if row["energy_joules"] is None:
            assert row["power_unavailable_reason"]
    symmetry = [r for r in rows if r["kind"] == "symmetry"]
    expected = {(s, seed, tuple(rank)) for s in config["strategies"]
                for seed in (config["permutation_seeds"][:1] if s == "identity" else config["permutation_seeds"])
                for rank in config["ranks"]}
    sweep = [r for r in rows if r["kind"] == "rank_sweep"]
    actual = {(r["strategy"], r["seed"], tuple(r["ranks"])) for r in sweep}
    assert actual == expected and len(sweep) == len(expected)
    assert {(r["strategy"], r["seed"]) for r in symmetry} == {(s, seed) for s, seed, _ in expected}
    for ranks in config["ranks"]:
        matched = [r for r in sweep if r["ranks"] == ranks]
        assert len({r["factorized_parameters"] for r in matched}) == 1
        expected_parameters = 3 * (prod(ranks) + sum(d * rank for d, rank in zip(config["shape"], ranks)))
        assert matched[0]["factorized_parameters"] == expected_parameters
    for row in sweep:
        assert row["quality_metrics"]["scored_tokens"] == config["quality"]["evaluation_tokens"] - 1
        assert row["quality_metrics"]["full_evaluation"] is False
        assert row["original_parameters"] == 3 * 6144 * 2048
        path = directory / "factors" / f"{row['strategy']}-{row['seed']}-{'x'.join(map(str, row['ranks']))}.safetensors"
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            assert 0 < header_size <= 65536
            header = json.loads(handle.read(header_size))
        total = 0
        for name in ("gate", "up", "down"):
            assert header[name + ".core"]["shape"] == row["ranks"]
            for axis, (size, rank) in enumerate(zip(row["shape"], row["ranks"])):
                assert header[f"{name}.factor{axis}"]["shape"] == [size, rank]
            for key, value in header.items():
                if key.startswith(name + "."):
                    assert value["dtype"] == "BF16"
                    start, end = value["data_offsets"]
                    total += (end - start) // 2
        assert total == row["factorized_parameters"]
    snapshot = directory / "source_snapshot.zip"
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == meta["source_snapshot_sha256"]
    with zipfile.ZipFile(snapshot) as archive:
        files = sorted(n for n in archive.namelist() if n.startswith("src/") and n.endswith(".py"))
        files += sorted(n for n in archive.namelist() if n.startswith("tests/") and n.endswith(".py"))
        files += ["pyproject.toml", "conftest.py", "requirements-lock.txt"]
        digest = hashlib.sha256()
        for name in files:
            digest.update(name.encode())
            digest.update(archive.read(name))
        assert digest.hexdigest() == meta["source_sha256"]
    history = directory.parent.parent / "experiments.jsonl"
    global_rows = [r for r in map(json.loads, history.read_text().splitlines()) if r["run_id"] == meta["run_id"]]
    assert global_rows == rows, "run raw rows and append-only global history diverged"
    return {"run_id": meta["run_id"], "acceptance": "passed", "baseline_configurations": len(baseline),
            "measured_generations": sum(r["iterations"] for r in baseline),
            "symmetry_configurations": len(symmetry), "rank_configurations": len(sweep),
            "bf16_artifacts_verified": len(sweep), "source_snapshot_verified": True,
            "raw_sha256": hashlib.sha256((directory / "raw.jsonl").read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    result = verify(args.run_directory)
    text = json.dumps(result, indent=2) + "\n"
    (args.run_directory / "VERIFICATION.json").write_text(text, encoding="utf-8")
    print(text)
