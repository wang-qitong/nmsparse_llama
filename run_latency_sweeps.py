#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SweepSpec:
    name: str
    sweep_sh: str
    result_json: str


def _run(cmd: List[str], *, env: Dict[str, str]) -> None:
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"command failed with exit code {p.returncode}: {' '.join(cmd)}")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_oom_sweep_points(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if result.get("oom_sweep", False):
        points = result.get("results", [])
        if not isinstance(points, list):
            raise TypeError("invalid sweep result: results must be a list")
        return points

    # Backward compatibility: if someone saved only a flat dict, try to infer.
    if "results" in result and isinstance(result["results"], list):
        return result["results"]

    raise ValueError("result json does not look like an OOM sweep output")


def _percentile(values: List[float], q: float) -> float:
    if len(values) == 0:
        raise ValueError("cannot compute percentile of empty list")
    if q < 0.0 or q > 100.0:
        raise ValueError(f"invalid percentile q={q}")

    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return float(xs[0])

    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = float(pos - lo)
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _aggregate(values: List[float], stat: str) -> float:
    if len(values) == 0:
        raise ValueError("cannot aggregate empty list")
    if stat == "mean":
        return float(sum(values) / len(values))
    if stat == "p50":
        return _percentile(values, 50.0)
    if stat == "p90":
        return _percentile(values, 90.0)
    if stat == "p99":
        return _percentile(values, 99.0)
    raise ValueError(f"unknown stat: {stat}")


def _print_latencies(
    points: List[Dict[str, Any]], *, header: str, stat: str
) -> List[Tuple[int, float, List[float]]]:
    rows: List[Tuple[int, float, List[float]]] = []
    print("=" * 80)
    print(header)
    print("=" * 80)

    for p in points:
        kvlen = int(p["kvlen"])
        all_ms = p.get("all_ms", [])
        if not isinstance(all_ms, list) or len(all_ms) == 0:
            raise ValueError(f"missing all_ms for kvlen={kvlen}")
        all_ms_f = [float(x) for x in all_ms]
        agg_ms = _aggregate(all_ms_f, stat)

        all_str = ", ".join(f"{x:.3f}" for x in all_ms_f)
        print(f"[LAT] kvlen={kvlen} n={len(all_ms_f)} {stat}={agg_ms:.3f}ms all_ms=[{all_str}]")
        rows.append((kvlen, agg_ms, all_ms_f))

    return rows


def _print_throughputs(
    points: List[Dict[str, Any]], *, header: str, batch: int, stat: str
) -> List[Tuple[int, float, List[float]]]:
    rows: List[Tuple[int, float, List[float]]] = []
    print("=" * 80)
    print(header)
    print("=" * 80)

    for p in points:
        kvlen = int(p["kvlen"])
        all_ms = p.get("all_ms", [])
        if not isinstance(all_ms, list) or len(all_ms) == 0:
            raise ValueError(f"missing all_ms for kvlen={kvlen}")
        all_ms_f = [float(x) for x in all_ms]
        all_tps_f = [float(batch) * 1000.0 / float(x) for x in all_ms_f]
        agg_tps = _aggregate(all_tps_f, stat)

        all_str = ", ".join(f"{x:.2f}" for x in all_tps_f)
        print(f"[TPS] kvlen={kvlen} n={len(all_tps_f)} {stat}={agg_tps:.2f} tok/s all_tps=[{all_str}]")
        rows.append((kvlen, agg_tps, all_tps_f))

    return rows


def _plot(
    rows_by_series: List[Tuple[str, List[Tuple[int, float, List[float]]]]], *, out_png: str, title: str, ylabel: str
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it via: pip install matplotlib\n"
            f"Original error: {e}"
        )

    plt.figure(figsize=(7.5, 5.0), dpi=160)
    for label, rows in rows_by_series:
        xs = [kv for (kv, mean, _) in rows]
        ys = [mean for (kv, mean, _) in rows]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=3, label=label)

    plt.title(title)
    plt.xlabel("sequence length (kvlen)")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"\n[Plot] saved: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run N:M OOM sweeps and plot latency-vs-kvlen.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "dense_pruned", "dense_original"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--kvlen_start", type=int, default=512)
    parser.add_argument("--kvlen_end", type=int, default=4096)
    parser.add_argument("--kvlen_step", type=int, default=128)
    parser.add_argument("--out_png", type=str, default="/wangqitong/latency_vs_kvlen.png")
    parser.add_argument("--title", type=str, default="LLaMA2-7B Decode Latency")
    parser.add_argument(
        "--plot_metric",
        type=str,
        default="latency_ms",
        choices=["latency_ms", "tokens_per_sec"],
        help="Metric to print/plot. tokens_per_sec uses batch*1000/latency_ms.",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default="mean",
        choices=["mean", "p50", "p90", "p99"],
        help="Aggregation over timed-forwards per kvlen point.",
    )
    parser.add_argument(
        "--include_dense_original",
        action="store_true",
        help="Also run an extra dense_original kvlen sweep (using 16:32 runner) and plot it together.",
    )
    parser.add_argument(
        "--dense_only",
        action="store_true",
        help="Only run dense kvlen sweep (using 16:32 runner) and plot it; skip all N:M specs.",
    )
    parser.add_argument(
        "--dense_mode",
        type=str,
        default="dense_original",
        choices=["dense_original", "dense_pruned"],
        help="Dense mode to use when --dense_only is set.",
    )
    parser.add_argument("--skip_run", action="store_true", help="Skip running sweeps and only parse/plot existing JSON")
    args = parser.parse_args()

    specs = [
        SweepSpec(
            name="2:4",
            sweep_sh="/wangqitong/2_4/run_oom_sweep.sh",
            result_json="/wangqitong/2_4/logs_mlp_2_4/sparse_result.json",
        ),
        SweepSpec(
            name="4:8",
            sweep_sh="/wangqitong/4_8/run_oom_sweep.sh",
            result_json="/wangqitong/4_8/logs_4_8/sparse_result.json",
        ),
        SweepSpec(
            name="16:32",
            sweep_sh="/wangqitong/16_32A800/run_oom_sweep.sh",
            result_json="/wangqitong/16_32A800/logs_16_32/sparse_result.json",
        ),
        SweepSpec(
            name="8:16",
            sweep_sh="/wangqitong/8_16/run_oom_sweep.sh",
            result_json="/wangqitong/8_16/logs_8_16/sparse_result.json",
        ),
        SweepSpec(
            name="32:64",
            sweep_sh="/wangqitong/32_64/run_oom_sweep.sh",
            result_json="/wangqitong/32_64/logs_32_64/sparse_result.json",
        ),
    ]

    env = dict(os.environ)
    env["E2E_OOM_SWEEP"] = "1"
    env["E2E_BATCH"] = str(int(args.batch))
    env["E2E_SEQ_LEN"] = str(int(args.seq_len))
    env["E2E_SWEEP_MAX_STEPS"] = str(int(args.max_steps))
    env["E2E_SWEEP_STEP"] = str(int(args.step))
    env["E2E_SWEEP_KVLEN_START"] = str(int(args.kvlen_start))
    env["E2E_SWEEP_KVLEN_END"] = str(int(args.kvlen_end))
    env["E2E_SWEEP_KVLEN_STEP"] = str(int(args.kvlen_step))

    # These are now mostly informational (we fixed the sweep to use 30 fwd/point in python),
    # but keep passing them for compatibility.
    env["E2E_SWEEP_REPEATS"] = "20"
    env["E2E_SWEEP_PRINT_ALL"] = "1"

    rows_by_series = []
    if not args.dense_only:
        for spec in specs:
            if not args.skip_run:
                if not os.path.exists(spec.sweep_sh):
                    raise FileNotFoundError(spec.sweep_sh)
                print("\n" + "#" * 80)
                print(f"[Run] {spec.name} sweep")
                print("#" * 80)
                _run(
                    [
                        "bash",
                        spec.sweep_sh,
                        args.mode,
                        str(args.gpu),
                        "20",
                        str(args.max_steps),
                        str(args.step),
                        str(args.kvlen_start),
                        str(args.kvlen_end),
                        str(args.kvlen_step),
                    ],
                    env=env,
                )

            if not os.path.exists(spec.result_json):
                raise FileNotFoundError(
                    f"result json not found for {spec.name}: {spec.result_json}\n"
                    "If you used --mode other than sparse, update the JSON path mapping in run_latency_sweeps.py."
                )

            result = _read_json(spec.result_json)
            points = _extract_oom_sweep_points(result)
            if args.plot_metric == "tokens_per_sec":
                rows = _print_throughputs(
                    points,
                    header=f"{spec.name} sweep throughputs (timed-forwards per point)",
                    batch=int(args.batch),
                    stat=args.stat,
                )
            else:
                rows = _print_latencies(
                    points,
                    header=f"{spec.name} sweep latencies (timed-forwards per point)",
                    stat=args.stat,
                )
            rows_by_series.append((spec.name, rows))

    if args.dense_only or args.include_dense_original:
        dense_mode = args.dense_mode if args.dense_only else "dense_original"
        dense_result_json = f"/wangqitong/16_32A800/logs_16_32/{dense_mode}_result.json"
        dense_spec = SweepSpec(
            name=dense_mode,
            sweep_sh="/wangqitong/16_32A800/run_oom_sweep.sh",
            result_json=dense_result_json,
        )

        if not args.skip_run:
            if not os.path.exists(dense_spec.sweep_sh):
                raise FileNotFoundError(dense_spec.sweep_sh)
            print("\n" + "#" * 80)
            print(f"[Run] {dense_mode} sweep")
            print("#" * 80)
            _run(
                [
                    "bash",
                    dense_spec.sweep_sh,
                    dense_mode,
                    str(args.gpu),
                    "20",
                    str(args.max_steps),
                    str(args.step),
                    str(args.kvlen_start),
                    str(args.kvlen_end),
                    str(args.kvlen_step),
                ],
                env=env,
            )

        if not os.path.exists(dense_spec.result_json):
            raise FileNotFoundError(f"result json not found for {dense_mode}: {dense_spec.result_json}")

        result = _read_json(dense_spec.result_json)
        points = _extract_oom_sweep_points(result)
        if args.plot_metric == "tokens_per_sec":
            rows = _print_throughputs(
                points,
                header=f"{dense_mode} sweep throughputs (timed-forwards per point)",
                batch=int(args.batch),
                stat=args.stat,
            )
        else:
            rows = _print_latencies(
                points,
                header=f"{dense_mode} sweep latencies (timed-forwards per point)",
                stat=args.stat,
            )
        rows_by_series.append((dense_spec.name, rows))

    if args.plot_metric == "latency_ms":
        ylabel = f"latency ({args.stat}) (ms)"
    else:
        ylabel = f"tokens/s ({args.stat})"

    _plot(rows_by_series, out_png=args.out_png, title=args.title, ylabel=ylabel)


if __name__ == "__main__":
    main()
