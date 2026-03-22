import argparse
import os
import time
from typing import Dict, List, Tuple

import pandas as pd
from transformers import AutoTokenizer

from gpu_utils import GPUMonitor, load_long_prompt, save_system_info
from inference_core import build_engine, run_e2e_request


WORKLOADS: Dict[str, Tuple[int, int]] = {
    "SS": (64, 64),
    "SL": (64, 512),
    "LS": (1024, 64),
    "LL": (1024, 512),
}


def parse_workloads(raw: str) -> List[str]:
    items = [x.strip().upper() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("--workloads cannot be empty")
    invalid = [x for x in items if x not in WORKLOADS]
    if invalid:
        raise ValueError(f"Invalid workloads: {invalid}. Available: {sorted(WORKLOADS.keys())}")
    return items


def build_prompt_for_tokens(tokenizer, base_text: str, target_tokens: int) -> str:
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_ids:
        base_ids = tokenizer.encode("hello world", add_special_tokens=False)

    ids: List[int] = []
    while len(ids) < target_tokens:
        ids.extend(base_ids)

    ids = ids[:target_tokens]
    return tokenizer.decode(ids, clean_up_tokenization_spaces=False)


def aggregate_mean(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    agg_map = {c: "mean" for c in value_cols}
    out = df.groupby(group_cols, as_index=False).agg(agg_map)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure TPJ for one locked GPU frequency and selected workloads")
    parser.add_argument("--workloads", type=str, default="SS,SL,LS,LL")
    parser.add_argument("--frequency-mhz", type=int, required=True,
                        help="Current locked GPU frequency label (set externally by bash)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--latency-slo-s", type=float, default=8.0)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    parser.add_argument("--append", action="store_true", help="Append rows to existing raw CSV")
    parser.add_argument("--model-path", type=str, default="./mistral_7b_model/LLM-Research/Mistral-7B-v0.3")
    parser.add_argument("--raw-out", type=str, default="./log/workload_tpj_freq_raw.csv")
    parser.add_argument("--summary-out", type=str, default="./log/workload_tpj_freq_summary.csv")
    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be >= 1")

    os.makedirs("./log", exist_ok=True)

    workloads = parse_workloads(args.workloads)
    frequency_mhz = int(args.frequency_mhz)

    save_system_info(args.model_path, script_name="workload_tpj_freq_scan")

    base_prompt = load_long_prompt()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompts: Dict[str, str] = {}
    for wl in workloads:
        in_len, _ = WORKLOADS[wl]
        prompts[wl] = build_prompt_for_tokens(tokenizer, base_prompt, in_len)

    engine = build_engine(model_path=args.model_path)

    from vllm import SamplingParams
    warmup_params = SamplingParams(temperature=0.0, max_tokens=8)
    engine.add_request(request_id="warmup", prompt="hello", params=warmup_params)
    while True:
        outs = engine.step()
        if any(o.request_id == "warmup" and o.finished for o in outs):
            break
    time.sleep(2)

    rows = []
    print(f"[*] Measuring at externally locked frequency: {frequency_mhz} MHz")
    for wl in workloads:
        in_len, out_len = WORKLOADS[wl]
        prompt = prompts[wl]
        for rep in range(1, args.repeat + 1):
            print(f"  - Running {wl} @ {frequency_mhz} MHz ({rep}/{args.repeat})")
            with GPUMonitor(interval=0.02) as monitor:
                req = run_e2e_request(engine, prompt=prompt, max_tokens=out_len)

            total_tokens = int(req["input_tokens"]) + int(req["output_tokens"])
            metrics = monitor.get_metrics(total_tokens)
            energy = float(metrics["total_energy_j"])
            tpj = (total_tokens / energy) if energy > 0 else 0.0

            row = {
                "workload": wl,
                "input_tokens_target": in_len,
                "output_tokens_target": out_len,
                "frequency_mhz": frequency_mhz,
                "repeat_idx": rep,
                "duration_s": float(metrics["duration_s"]),
                "ttft_s": float(req["ttft_s"]),
                "tpot_s": float(req["tpot_s"]),
                "avg_power_w": float(metrics["avg_power_w"]),
                "peak_power_w": float(metrics["peak_power_w"]),
                "total_energy_j": energy,
                "throughput_tps": float(metrics["throughput_tps"]),
                "j_per_token": float(metrics["j_per_token"]),
                "tpj": round(tpj, 6),
                "input_tokens_actual": int(req["input_tokens"]),
                "output_tokens_actual": int(req["output_tokens"]),
                "slo_s": float(args.latency_slo_s),
                "slo_met": 1 if float(metrics["duration_s"]) <= float(args.latency_slo_s) else 0,
            }
            rows.append(row)

            time.sleep(args.cooldown_s)

    raw_df = pd.DataFrame(rows)
    write_mode = "a" if args.append and os.path.exists(args.raw_out) else "w"
    write_header = not (args.append and os.path.exists(args.raw_out))
    raw_df.to_csv(args.raw_out, mode=write_mode, header=write_header, index=False)

    full_raw_df = pd.read_csv(args.raw_out)

    value_cols = [
        "duration_s",
        "ttft_s",
        "tpot_s",
        "avg_power_w",
        "peak_power_w",
        "total_energy_j",
        "throughput_tps",
        "j_per_token",
        "tpj",
        "input_tokens_actual",
        "output_tokens_actual",
        "slo_met",
    ]
    summary_df = aggregate_mean(
        full_raw_df,
        group_cols=["workload", "input_tokens_target", "output_tokens_target", "frequency_mhz", "slo_s"],
        value_cols=value_cols,
    )
    summary_df = summary_df.rename(columns={"slo_met": "slo_met_rate"})
    summary_df.to_csv(args.summary_out, index=False)

    print("\n==========================================================")
    print(f"Raw data saved to: {args.raw_out}")
    print(f"Summary saved to:  {args.summary_out}")
    print("Tip: run plot/plot_workload_tpj_vs_freq.py for figures.")
    print("==========================================================")


if __name__ == "__main__":
    main()
