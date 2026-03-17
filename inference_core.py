import inspect
import os
import time
import argparse
import uuid
import vllm
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from gpu_utils import load_long_prompt, save_system_info, extract_ttft_tpot, GPUMonitor
# ---------------------------------------------------------------------------
# E2E 推理辅助：使用 LLMEngine.step() 精确捕获 TTFT
# ---------------------------------------------------------------------------

def build_engine(model_path: str, max_num_seqs: int = 32, max_num_batched_tokens: int = 8192) -> LLMEngine:
    engine_kwargs = {
        "model": model_path,
        "enforce_eager": True,
        "enable_chunked_prefill": False,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
    }
    arg_names = inspect.signature(EngineArgs).parameters
    if "disable_log_requests" in arg_names:
        engine_kwargs["disable_log_requests"] = True
    if "max_model_len" in arg_names:
        engine_kwargs["max_model_len"] = max_num_batched_tokens
    return LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))


def run_e2e_request(engine: LLMEngine, prompt: str, max_tokens: int = 256) -> dict:
    """通过 step() 循环精确测量 TTFT，返回耗时统计信息。"""
    request_id = f"req-{uuid.uuid4()}"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    engine.add_request(request_id=request_id, prompt=prompt, params=sampling_params)

    t0 = time.perf_counter()
    ttft_s = None
    final_output = None

    while True:
        step_outputs = engine.step()
        now = time.perf_counter()

        for out in step_outputs:
            if out.request_id != request_id:
                continue
            generated_len = len(out.outputs[0].token_ids) if out.outputs else 0
            if ttft_s is None and generated_len > 0:
                ttft_s = now - t0
            if out.finished:
                final_output = out
                break

        if final_output is not None:
            break

    total_duration_s = time.perf_counter() - t0
    if ttft_s is None:
        ttft_s = total_duration_s

    output_tokens = len(final_output.outputs[0].token_ids)
    tpot_s = max(total_duration_s - ttft_s, 0.0) / max(output_tokens, 1)

    return {
        "ttft_s": round(ttft_s, 4),
        "total_duration_s": round(total_duration_s, 4),
        "tpot_s": round(tpot_s, 6),
        "output_tokens": output_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["4a", "4a_e2e", "4b"])
    parser.add_argument("--freq", type=int, default=0)
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()

    # 1. 长 prompt 读取本地文件（Task 1）
    prefill_prompt = load_long_prompt()
    decode_prompt = "Hello."

    # 2. 初始化模型路径和系统信息
    model_path = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3"
    save_system_info(model_path, script_name="inference_core")

    # 3. 执行对应的 Task
    os.makedirs("./log", exist_ok=True)

    if args.task in ("4a", "4b"):
        # 高层 LLM API：4a（PD 分别测试）和 4b（Batch 测试）共用
        llm = LLM(model=model_path, enforce_eager=True)
        llm.generate([decode_prompt], SamplingParams(max_tokens=10), use_tqdm=False)
        time.sleep(2)

    elif args.task == "4a_e2e":
        # 低层 LLMEngine API：用 step() 精确捕获 TTFT 的完整推理
        engine = build_engine(model_path)
        # 预热
        warmup_params = SamplingParams(temperature=0.0, max_tokens=8)
        engine.add_request(request_id="warmup", prompt="hello", params=warmup_params)
        while True:
            outs = engine.step()
            if any(o.request_id == "warmup" and o.finished for o in outs):
                break
        time.sleep(2)

    if args.task == "4a":
        # === Prefill 阶段测试 ===
        # 使用长prompt，max_tokens=1，专注测量TTFT
        prefill_params = SamplingParams(temperature=0.0, max_tokens=1)
        start_pre = time.perf_counter()
        with GPUMonitor() as monitor:
            out = llm.generate([prefill_prompt], prefill_params, use_tqdm=False)
        pre_duration = time.perf_counter() - start_pre
        in_tokens = len(out[0].prompt_token_ids)
        pre_out_tokens = len(out[0].outputs[0].token_ids)
        m_pre = monitor.get_metrics(in_tokens)
        # 对于Prefill阶段，TTFT更重要，因为它主要受频率影响
        pre_ttft_s, pre_tpot_s = extract_ttft_tpot(out[0], pre_duration, pre_out_tokens)

        time.sleep(2)

        # === Decode 阶段测试 ===
        # 使用短prompt，生成更多tokens，专注测量TPOT
        decode_params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
        start_dec = time.perf_counter()
        with GPUMonitor() as monitor:
            out = llm.generate([decode_prompt], decode_params, use_tqdm=False)
        dec_duration = time.perf_counter() - start_dec
        out_tokens = len(out[0].outputs[0].token_ids)
        m_dec = monitor.get_metrics(out_tokens)
        # 对于Decode阶段，TPOT更重要，因为它反映了持续生成的速度
        dec_ttft_s, dec_tpot_s = extract_ttft_tpot(out[0], dec_duration, out_tokens)

        # 保存结果到 CSV
        with open("./log/task4a_results.csv", "a") as f:
            f.write(
                f"Prefill,{args.freq},{m_pre['duration_s']},{pre_ttft_s},{pre_tpot_s},"
                f"{m_pre['avg_power_w']},{m_pre['peak_power_w']},{m_pre['total_energy_j']},{m_pre['throughput_tps']},"
                f"{m_pre['j_per_token']},{pre_out_tokens}\n"
            )
            f.write(
                f"Decode,{args.freq},{m_dec['duration_s']},{dec_ttft_s},{dec_tpot_s},"
                f"{m_dec['avg_power_w']},{m_dec['peak_power_w']},{m_dec['total_energy_j']},{m_dec['throughput_tps']},"
                f"{m_dec['j_per_token']},{out_tokens}\n"
            )

    elif args.task == "4a_e2e":
        # === 完整 E2E 推理测试（long prompt + 256 decode tokens）===
        e2e_params_max_tokens = 256
        with GPUMonitor() as monitor:
            e2e_result = run_e2e_request(engine, prefill_prompt, max_tokens=e2e_params_max_tokens)
        m_e2e = monitor.get_metrics(e2e_result["output_tokens"])

        with open("./log/task4a_e2e_results.csv", "a") as f:
            f.write(
                f"E2E,{args.freq},{m_e2e['duration_s']},{e2e_result['ttft_s']},{e2e_result['tpot_s']},"
                f"{m_e2e['avg_power_w']},{m_e2e['peak_power_w']},{m_e2e['total_energy_j']},{m_e2e['throughput_tps']},"
                f"{m_e2e['j_per_token']},{e2e_result['output_tokens']}\n"
            )

    elif args.task == "4b":
        # === Batch Size 测试 ===
        batch_params = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
        prompts = [prefill_prompt] * args.bs  # 复制长输入构建 Batch
        
        start_batch = time.perf_counter()
        with GPUMonitor() as monitor:
            outs = llm.generate(prompts, batch_params, use_tqdm=False)
        batch_duration = time.perf_counter() - start_batch
        
        total_tokens = sum([len(o.outputs[0].token_ids) for o in outs])
        m_batch = monitor.get_metrics(total_tokens)

        # Batch 模式下做平均 TTFT/TPOT，便于横向比较。
        ttft_list = []
        for o in outs:
            o_tokens = len(o.outputs[0].token_ids)
            ttft_s, _ = extract_ttft_tpot(o, batch_duration, o_tokens)
            ttft_list.append(ttft_s)

        mean_ttft_s = round(sum(ttft_list) / len(ttft_list), 4) if ttft_list else round(batch_duration, 4)
        mean_tpot_s = round(max(batch_duration - mean_ttft_s, 0.0) / max(total_tokens, 1), 6)

        # 保存结果到 CSV
        with open("./log/task4b_results.csv", "a") as f:
            f.write(
                f"{args.bs},{m_batch['duration_s']},{mean_ttft_s},{mean_tpot_s},{m_batch['avg_power_w']},{m_batch['peak_power_w']},"
                f"{m_batch['total_energy_j']},{m_batch['throughput_tps']},{m_batch['j_per_token']},{total_tokens}\n"
            )

if __name__ == "__main__":
    main()
