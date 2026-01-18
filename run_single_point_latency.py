#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Spec:
    name: str
    sweep_sh: str
    result_json: str
    mode: str


def _run(cmd: List[str], *, env: Dict[str, str]) -> None:
    p = subprocess.run(cmd, env=env, stdout=None, stderr=None)
    if p.returncode != 0:
        raise RuntimeError(f"command failed with code={p.returncode}: {' '.join(cmd)}")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_single_point_mean_ms(result: Dict[str, Any], *, kvlen: int) -> float:
    if result.get("oom_sweep", False):
        points = result.get("results", [])
        if not isinstance(points, list):
            raise TypeError("invalid sweep result: results must be a list")
        for p in points:
            if int(p.get("kvlen", -1)) == int(kvlen):
                all_ms = p.get("all_ms", [])
                if not isinstance(all_ms, list) or len(all_ms) == 0:
                    raise ValueError(f"missing all_ms at kvlen={kvlen}")
                all_ms_f = [float(x) for x in all_ms]
                return float(sum(all_ms_f) / len(all_ms_f))
        raise ValueError(f"kvlen={kvlen} not found in sweep results")

    if "kvlen" in result and int(result.get("kvlen")) == int(kvlen):
        if "ms_per_token" in result:
            return float(result["ms_per_token"])
        if "total_ms" in result:
            return float(result["total_ms"])

    raise ValueError("result json does not look like an OOM sweep output")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-point latency for all patterns + dense.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--kvlen", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--use_cuda_graph", action="store_true")
    parser.add_argument("--only_16_32_dense", action="store_true")
    parser.add_argument("--skip_run", action="store_true")
    args = parser.parse_args()

    kvlen = int(args.kvlen)
    repeats = max(1, int(args.repeats))

    if args.only_16_32_dense:
        specs: List[Spec] = [
            Spec(
                name="16:32",
                sweep_sh="/wangqitong/16_32A800/run_oom_sweep.sh",
                result_json="/wangqitong/16_32A800/logs_16_32/sparse_result.json",
                mode="sparse",
            ),
            Spec(
                name="dense_original",
                sweep_sh="/wangqitong/16_32A800/run_oom_sweep.sh",
                result_json="/wangqitong/16_32A800/logs_16_32/dense_original_result.json",
                mode="dense_original",
            ),
        ]
    else:
        specs = [
            Spec(
                name="2:4",
                sweep_sh="/wangqitong/2_4/run_oom_sweep.sh",
                result_json="/wangqitong/2_4/logs_mlp_2_4/sparse_result.json",
                mode="sparse",
            ),
            Spec(
                name="4:8",
                sweep_sh="/wangqitong/4_8/run_oom_sweep.sh",
                result_json="/wangqitong/4_8/logs_4_8/sparse_result.json",
                mode="sparse",
            ),
            Spec(
                name="8:16",
                sweep_sh="/wangqitong/8_16/run_oom_sweep.sh",
                result_json="/wangqitong/8_16/logs_8_16/sparse_result.json",
                mode="sparse",
            ),
            Spec(
                name="16:32",
                sweep_sh="/wangqitong/16_32A800/run_oom_sweep.sh",
                result_json="/wangqitong/16_32A800/logs_16_32/sparse_result.json",
                mode="sparse",
            ),
            Spec(
                name="32:64",
                sweep_sh="/wangqitong/32_64/run_oom_sweep.sh",
                result_json="/wangqitong/32_64/logs_32_64/sparse_result.json",
                mode="sparse",
            ),
            Spec(
                name="dense_original",
                sweep_sh="/wangqitong/16_32A800/run_oom_sweep.sh",
                result_json="/wangqitong/16_32A800/logs_16_32/dense_original_result.json",
                mode="dense_original",
            ),
        ]

    env = dict(os.environ)
    env["E2E_OOM_SWEEP"] = "1"
    env["E2E_BATCH"] = str(int(args.batch))
    env["E2E_SEQ_LEN"] = str(int(args.seq_len))
    env["E2E_SWEEP_REPEATS"] = str(repeats)
    env["E2E_SWEEP_PRINT_ALL"] = "1"
    env["E2E_SWEEP_MAX_STEPS"] = str(int(kvlen))
    env["E2E_SWEEP_STEP"] = "1"
    env["E2E_SWEEP_KVLEN_START"] = str(int(kvlen))
    env["E2E_SWEEP_KVLEN_END"] = str(int(kvlen))
    env["E2E_SWEEP_KVLEN_STEP"] = "1"
    if args.use_cuda_graph:
        env["E2E_USE_CUDA_GRAPH"] = "1"
    env["PATH"] = f"/wangqitong/miniconda3/envs/myenv/bin:{env.get('PATH','')}"
    env["CONDA_PREFIX"] = "/wangqitong/miniconda3/envs/myenv"

    rows: List[Tuple[str, Optional[float], Optional[str]]] = []

    for s in specs:
        if not args.skip_run:
            if not os.path.exists(s.sweep_sh):
                raise FileNotFoundError(s.sweep_sh)
            _run(
                [
                    "bash",
                    s.sweep_sh,
                    s.mode,
                    str(args.gpu),
                    str(repeats),
                    str(kvlen),
                    "1",
                    str(kvlen),
                    str(kvlen),
                    "1",
                ],
                env=env,
            )

        if not os.path.exists(s.result_json):
            rows.append((s.name, None, f"missing json: {s.result_json}"))
            continue

        try:
            j = _read_json(s.result_json)
            mean_ms = _extract_single_point_mean_ms(j, kvlen=kvlen)
            rows.append((s.name, mean_ms, None))
        except Exception as e:
            rows.append((s.name, None, str(e)))

    print("\n" + "=" * 80)
    print(f"Single-point decode latency @ kvlen={kvlen} batch={args.batch} seq_len={args.seq_len} (lower is better)")
    print("=" * 80)

    ok = [(n, ms) for (n, ms, err) in rows if ms is not None and err is None]
    bad = [(n, err) for (n, ms, err) in rows if err is not None]

    for name, ms in sorted(ok, key=lambda x: x[1]):
        tps = float(args.batch) * 1000.0 / float(ms)
        print(f"{name:14s}  mean_ms={ms:9.3f}  tokens/s={tps:9.2f}")

    if bad:
        print("\n[Errors]")
        for name, err in bad:
            print(f"{name:14s}  {err}")


if __name__ == "__main__":
    main()
