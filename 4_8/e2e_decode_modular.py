#\!/usr/bin/env python3
"""
端到端 Sparse Decode 验证 - 模块化版本
支持通过环境变量 E2E_MODE 指定测试模式：sparse, dense_pruned, dense_original
每次只运行一种配置，进程退出后显存完全回收
"""

import os
import sys
import time
import json
import threading

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print(
    f"[TF32] matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
    f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}"
)
from transformers import LlamaForCausalLM, AutoTokenizer
from torch.utils.cpp_extension import load
import torch.cuda.nvtx as nvtx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nmsparse_test_utils import load_and_prune_weight, convert_to_sparse_format

LLAMA_MODEL_PATH = "/wangqitong/llama_2-7b"
DEVICE = "cuda"
DTYPE = torch.float32

PROMPT_TEXT = "The future of artificial intelligence is"
NUM_DECODE_STEPS = int(os.getenv("E2E_STEPS", "1"))
NUM_WARMUP_DECODE = 0
NUM_TEST_ITERATIONS = int(os.getenv("E2E_ITERS", "100"))
BATCH_SIZE = int(os.getenv("E2E_BATCH", "8"))
SEQ_LEN = int(os.getenv("E2E_SEQ_LEN", "512"))

OOM_SWEEP = os.getenv("E2E_OOM_SWEEP", "0") == "1"
SWEEP_REPEATS = int(os.getenv("E2E_SWEEP_REPEATS", "20"))
SWEEP_MAX_STEPS = int(os.getenv("E2E_SWEEP_MAX_STEPS", "4096"))
SWEEP_STEP = int(os.getenv("E2E_SWEEP_STEP", "1"))

SWEEP_KVLEN_START = int(os.getenv("E2E_SWEEP_KVLEN_START", "0"))
SWEEP_KVLEN_END = int(os.getenv("E2E_SWEEP_KVLEN_END", "0"))
SWEEP_KVLEN_STEP = int(os.getenv("E2E_SWEEP_KVLEN_STEP", "0"))

SPARSITY = 0.5
NUM_BANK_VAL = 8
NUM_THREADS = 128
PROFILE_COMPONENTS = False

TEST_MODE = os.getenv("E2E_MODE", "sparse")
if TEST_MODE not in ["sparse", "dense_pruned", "dense_original"]:
    print(f"错误：无效的 E2E_MODE={TEST_MODE}")
    print("有效值: sparse, dense_pruned, dense_original")
    sys.exit(1)

print("=" * 80)
print(f"端到端 Sparse Decode 验证 - 模式: {TEST_MODE.upper()}")
print("=" * 80)
print(f"  模型: {LLAMA_MODEL_PATH}")
print(f"  Batch: {BATCH_SIZE}, Seq: {SEQ_LEN}")
print(f"  Decode steps: {NUM_DECODE_STEPS}")
print(f"  测试迭代次数: {NUM_TEST_ITERATIONS}")
print(f"  稀疏率: {SPARSITY} (50%)")
print(f"  稀疏化范围: 所有 32 层的 QKVO + MLP (7个投影/层, 4:8稀疏模式)")
print(f"  Profile components: {PROFILE_COMPONENTS}")
print("=" * 80)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

print("\n加载 nmSPARSE kernel...")
script_dir = os.path.dirname(os.path.abspath(__file__))
wrapper_cu_path = os.path.join(script_dir, "nmsparse_wrapper_with_initialdata.cu")

if not os.path.exists(wrapper_cu_path):
    print(f"✗ 找不到 CUDA wrapper: {wrapper_cu_path}")
    sys.exit(1)

try:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    cuda_include_path = f"{conda_prefix}/targets/x86_64-linux/include"

    nmsparse_module = load(
        name="nmsparse_fig9_4_8",
        sources=[wrapper_cu_path],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-I{cuda_include_path}"],
        verbose=True,
    )
    print("✓ nmSPARSE kernel 编译/加载成功")
except Exception as e:
    print(f"✗ Kernel 编译/加载失败: {e}")
    sys.exit(1)

print(f"\n加载模型: {LLAMA_MODEL_PATH}")
model = LlamaForCausalLM.from_pretrained(
    LLAMA_MODEL_PATH,
    torch_dtype=DTYPE,
    device_map=DEVICE,
)
model.eval()
num_layers = len(model.model.layers)
print(f"✓ 模型加载完成，共 {num_layers} 层")

print("\n加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
print("✓ Tokenizer 加载完成")

_ATTENTION_CTX = threading.local()

pruned_cache_file = "/wangqitong/sparse_params_sparsity0.5_bank8_mlp_qkvo_4_8.pt"

if os.path.exists(pruned_cache_file):
    print(f"\n从缓存加载剪枝参数: {pruned_cache_file}")
    sparse_params = torch.load(pruned_cache_file, map_location="cpu")
    print("✓ 剪枝参数加载完成")
    for layer_idx in range(num_layers):
        for proj_name in sparse_params[layer_idx].keys():
            entry = sparse_params[layer_idx][proj_name]
            K = entry["K"]
            N = entry["N"]
            num_bank = K // NUM_BANK_VAL
            num_nonzeros_per_bank = int(NUM_BANK_VAL * (1 - SPARSITY))
            entry["w"] = num_bank * num_nonzeros_per_bank
            entry["h"] = N
            entry["block_width"] = num_nonzeros_per_bank
            entry["vec_width"] = NUM_BANK_VAL
else:
    print("\n缓存不存在，开始剪枝所有 32 层的 QKVO + MLP...")
    sparse_params = {}
    for layer_idx in range(num_layers):
        sparse_params[layer_idx] = {}
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            real_weight, pruned_weight, pruned_mask, K, N = load_and_prune_weight(
                model, layer_idx, proj_name, SPARSITY, NUM_BANK_VAL
            )
            mat_data_cpu, mat_index_cpu = convert_to_sparse_format(pruned_weight, pruned_mask, NUM_BANK_VAL)

            num_bank = K // NUM_BANK_VAL
            num_nonzeros_per_bank = int(NUM_BANK_VAL * (1 - SPARSITY))
            w = num_bank * num_nonzeros_per_bank
            h = N

            sparse_params[layer_idx][proj_name] = {
                "mat_data": mat_data_cpu,
                "mat_index": mat_index_cpu,
                "pruned_weight": pruned_weight,
                "original_weight": real_weight,
                "w": w,
                "h": h,
                "block_width": num_nonzeros_per_bank,
                "vec_width": NUM_BANK_VAL,
                "K": K,
                "N": N,
            }

        if (layer_idx + 1) % 4 == 0:
            print(f"  ✓ Layer {layer_idx} 剪枝完成 (7 projections)")

    os.makedirs(os.path.dirname(pruned_cache_file), exist_ok=True)
    torch.save(sparse_params, pruned_cache_file)
    print(f"✓ 剪枝完成并保存到 {pruned_cache_file}")

print("\n将稀疏参数移到 GPU...")
for layer_idx in range(num_layers):
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        sparse_params[layer_idx][proj_name]["mat_data"] = sparse_params[layer_idx][proj_name]["mat_data"].to(DEVICE)
        sparse_params[layer_idx][proj_name]["mat_index"] = sparse_params[layer_idx][proj_name]["mat_index"].to(DEVICE)
print("✓ 稀疏参数已移到 GPU")


class SparsifiedLinear:
    def __init__(self, original_module, nmsparse_module, sparse_param, name, timer_events=None):
        self.original_module = original_module
        self.nmsparse_module = nmsparse_module
        self.name = name
        self._timer_events = timer_events

        self.mat_data = sparse_param["mat_data"]
        self.mat_index = sparse_param["mat_index"]
        self.w = sparse_param["w"]
        self.h = sparse_param["h"]
        self.block_width = sparse_param["block_width"]
        self.vec_width = sparse_param["vec_width"]
        self.bias = original_module.bias

    def __call__(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_2d = x.reshape(-1, hidden_dim)
        minibatch = x_2d.shape[0]
        vecNum = x_2d.shape[1]

        nvtx.range_push("CPP::nmsparse_forward")
        output = self.nmsparse_module.forward(
            x_2d,
            self.mat_data,
            self.mat_index,
            self.w,
            self.h,
            self.block_width,
            NUM_THREADS,
            self.vec_width,
            minibatch,
            vecNum,
        )
        nvtx.range_pop()

        if self.bias is not None:
            output = output + self.bias

        output = output.contiguous().view(batch_size, seq_len, -1)
        return output


class _TimedForward:
    def __init__(self, original_forward, events_list, tag="PY::dense_linear"):
        self.original_forward = original_forward
        self.events_list = events_list
        self.tag = tag

    def __call__(self, *args, **kwargs):
        se, ee = None, None
        if self.events_list is not None:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
        result = self.original_forward(*args, **kwargs)
        if self.events_list is not None:
            ee.record()
            self.events_list.append((se, ee))
        return result


original_forwards = {}
original_block_forwards = {}
for layer_idx in range(num_layers):
    original_forwards[layer_idx] = {}
    layer = model.model.layers[layer_idx]
    original_block_forwards[layer_idx] = {
        "self_attn": layer.self_attn.forward,
        "mlp": layer.mlp.forward,
    }
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        proj_module = getattr(layer.self_attn, proj_name)
        original_forwards[layer_idx][proj_name] = proj_module.forward
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        proj_module = getattr(layer.mlp, proj_name)
        original_forwards[layer_idx][proj_name] = proj_module.forward


def get_projection_module(layer, proj_name):
    if proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        return getattr(layer.self_attn, proj_name)
    if proj_name in ["gate_proj", "up_proj", "down_proj"]:
        return getattr(layer.mlp, proj_name)
    raise ValueError(f"未知的投影名称: {proj_name}")


print("\nTokenize 输入...")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

inputs = tokenizer(
    [PROMPT_TEXT] * BATCH_SIZE,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=SEQ_LEN,
)
input_ids = inputs["input_ids"].to(DEVICE)
attention_mask_tok = inputs["attention_mask"].to(DEVICE)
batch_size = input_ids.shape[0]
print(f"✓ Tokenize 完成: batch={batch_size}, seq={input_ids.shape[1]}")


def _resize_past_key_values(past_key_values, target_kv_len: int):
    if past_key_values is None:
        return None

    resized = []
    for layer in past_key_values:
        if layer is None:
            resized.append(None)
            continue

        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError(f"unsupported past_key_values layer type: {type(layer)}")

        k, v = layer[0], layer[1]
        if not (torch.is_tensor(k) and torch.is_tensor(v)):
            raise TypeError("past_key_values must contain tensors")

        cur_len = int(k.size(-2))
        if cur_len == target_kv_len:
            resized.append((k, v))
            continue

        if cur_len > target_kv_len:
            k2 = k[..., :target_kv_len, :].contiguous()
            v2 = v[..., :target_kv_len, :].contiguous()
            resized.append((k2, v2))
            continue

        pad_len = int(target_kv_len - cur_len)
        k_pad = torch.zeros((*k.shape[:-2], pad_len, k.shape[-1]), device=k.device, dtype=k.dtype)
        v_pad = torch.zeros((*v.shape[:-2], pad_len, v.shape[-1]), device=v.device, dtype=v.dtype)
        k2 = torch.cat([k, k_pad], dim=-2).contiguous()
        v2 = torch.cat([v, v_pad], dim=-2).contiguous()
        resized.append((k2, v2))

    return tuple(resized)


def _oom_sweep_decode(*, base_past_key_values, base_attention_mask, base_input_ids, batch_size: int):
    import numpy as np

    print_all = os.getenv("E2E_SWEEP_PRINT_ALL", "0") == "1"

    warmup_forwards = 10
    timed_forwards = 20

    print("\n[OOM-SWEEP] 开始逐步增加 decode token 数，直到 OOM...")
    print(f"  - forwards/step: {warmup_forwards + timed_forwards} (warmup={warmup_forwards}, timed={timed_forwards})")
    if SWEEP_KVLEN_START > 0 and SWEEP_KVLEN_END > 0 and SWEEP_KVLEN_STEP > 0:
        print(f"  - kvlen range:  [{SWEEP_KVLEN_START}, {SWEEP_KVLEN_END}] step={SWEEP_KVLEN_STEP}")
        sweep_points = [(kvlen, kvlen - 1) for kvlen in range(SWEEP_KVLEN_START, SWEEP_KVLEN_END + 1, SWEEP_KVLEN_STEP)]
    else:
        print(f"  - max steps:    {SWEEP_MAX_STEPS}")
        print(f"  - step size:    {SWEEP_STEP}")
        kvlen_before = int(base_attention_mask.size(1))
        sweep_points = []
        for step_idx in range(SWEEP_STEP, SWEEP_MAX_STEPS + 1, SWEEP_STEP):
            kvlen_total = int(kvlen_before + step_idx)
            sweep_points.append((kvlen_total, kvlen_total - 1))

    results = []
    last_ok_kvlen = 0

    for kvlen_total, target_past_len in sweep_points:
        times_ms = []
        try:
            sweep_past = _resize_past_key_values(base_past_key_values, target_past_len)
            sweep_attn_mask = torch.ones(
                (batch_size, target_past_len + 1),
                device=base_attention_mask.device,
                dtype=base_attention_mask.dtype,
            )

            with torch.no_grad():
                for _ in range(warmup_forwards):
                    _ = model(
                        input_ids=base_input_ids,
                        past_key_values=sweep_past,
                        attention_mask=sweep_attn_mask,
                        use_cache=True,
                        return_dict=True,
                    )

                for _ in range(timed_forwards):
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    outputs = model(
                        input_ids=base_input_ids,
                        past_key_values=sweep_past,
                        attention_mask=sweep_attn_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                    end_event.record()
                    del outputs
                    torch.cuda.synchronize()
                    times_ms.append(float(start_event.elapsed_time(end_event)))
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda out of memory" in msg:
                torch.cuda.empty_cache()
                print(f"\n[OOM-SWEEP] OOM at kvlen={kvlen_total}")
                print(f"[OOM-SWEEP] last ok kvlen={last_ok_kvlen}")
                break
            raise

        last_ok_kvlen = kvlen_total
        mean_ms = float(np.mean(times_ms))
        p50_ms = float(np.percentile(times_ms, 50))
        p90_ms = float(np.percentile(times_ms, 90))
        p99_ms = float(np.percentile(times_ms, 99))

        if print_all:
            times_str = ", ".join(f"{t:.3f}" for t in times_ms)
            print(f"[OOM-SWEEP] kvlen={kvlen_total:5d} all_ms=[{times_str}]")

        print(
            f"[OOM-SWEEP] kvlen={kvlen_total:5d} "
            f"mean={mean_ms:.3f}ms p50={p50_ms:.3f}ms p90={p90_ms:.3f}ms p99={p99_ms:.3f}ms"
        )
        results.append(
            {
                "step": kvlen_total,
                "kvlen": kvlen_total,
                "repeats": timed_forwards,
                "mean_ms": mean_ms,
                "p50_ms": p50_ms,
                "p90_ms": p90_ms,
                "p99_ms": p99_ms,
                "all_ms": times_ms,
            }
        )

    return {"oom_sweep": True, "last_ok_step": last_ok_kvlen, "results": results}


def test_sparse():
    print("\n" + "=" * 80)
    print("测试: SPARSE (剪枝后权重 + nmSPARSE kernel)")
    print("=" * 80)

    print("\n[Prefill] 加载剪枝后权重（dense 格式）...")
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            proj_module = get_projection_module(layer, proj_name)
            pruned_weight = sparse_params[layer_idx][proj_name]["pruned_weight"]
            proj_module.weight.data = pruned_weight.T.to(DEVICE)
    print("✓ 剪枝后权重已加载")

    print("\n[Prefill] 生成 KV cache...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        torch.cuda.synchronize()
        start_event.record()
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask_tok,
            use_cache=True,
            return_dict=True,
        )
        end_event.record()
        torch.cuda.synchronize()

    prefill_time = start_event.elapsed_time(end_event)
    past_key_values = outputs.past_key_values
    attention_mask = attention_mask_tok
    del outputs
    torch.cuda.empty_cache()

    print(f"✓ Prefill 完成: {prefill_time:.3f} ms")

    last_indices = attention_mask.sum(dim=1) - 1
    cur_input_ids = input_ids.gather(1, last_indices.unsqueeze(1))

    print("\n[Decode] 应用稀疏化...")
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            proj_module = get_projection_module(layer, proj_name)
            proj_module.forward = SparsifiedLinear(
                proj_module,
                nmsparse_module,
                sparse_params[layer_idx][proj_name],
                f"layer{layer_idx}.{proj_name}",
                None,
            )

        layer.self_attn.forward = original_block_forwards[layer_idx]["self_attn"]
        layer.mlp.forward = original_block_forwards[layer_idx]["mlp"]

    print("✓ 稀疏化已应用")

    if OOM_SWEEP:
        return (
            _oom_sweep_decode(
                base_past_key_values=past_key_values,
                base_attention_mask=attention_mask.clone(),
                base_input_ids=cur_input_ids.clone(),
                batch_size=batch_size,
            ),
            [],
        )

    print(f"\n[Decode] 运行 {NUM_TEST_ITERATIONS} 次测试（每次 {NUM_DECODE_STEPS} steps）...")
    all_times_ms = []
    all_wall_times_ms = []
    tokens = []

    with torch.no_grad():
        for test_iter in range(NUM_TEST_ITERATIONS):
            decode_past = past_key_values
            decode_attn_mask = attention_mask.clone()
            decode_input_ids = cur_input_ids.clone()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start_event.record()

            for _ in range(NUM_DECODE_STEPS):
                new_token_mask = torch.ones((batch_size, 1), device=decode_attn_mask.device, dtype=decode_attn_mask.dtype)
                cur_attn_mask = torch.cat([decode_attn_mask, new_token_mask], dim=1)
                outputs = model(
                    input_ids=decode_input_ids,
                    past_key_values=decode_past,
                    attention_mask=cur_attn_mask,
                    use_cache=True,
                    return_dict=True,
                )
                decode_past = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                if test_iter == 0:
                    tokens.append(next_token.detach())
                decode_input_ids = next_token.unsqueeze(-1)
                decode_attn_mask = cur_attn_mask

            end_event.record()
            torch.cuda.synchronize()
            iter_time_ms = start_event.elapsed_time(end_event)
            iter_wall_ms = (time.perf_counter() - t0) * 1000.0
            all_times_ms.append(iter_time_ms)
            all_wall_times_ms.append(iter_wall_ms)

            if (test_iter + 1) % 10 == 0:
                print(f"  完成 {test_iter + 1}/{NUM_TEST_ITERATIONS} 次测试")

    import numpy as np

    total_ms = np.mean(all_times_ms)
    wall_ms = np.mean(all_wall_times_ms)
    ms_per_token = total_ms / NUM_DECODE_STEPS
    tokens_per_sec = batch_size * 1000.0 / ms_per_token

    result = {
        "mode": "sparse",
        "total_ms": float(total_ms),
        "wall_ms": float(wall_ms),
        "ms_per_token": float(ms_per_token),
        "tokens_per_sec": float(tokens_per_sec),
        "batch_size": int(batch_size),
        "seq_len": int(SEQ_LEN),
        "num_decode_steps": int(NUM_DECODE_STEPS),
        "num_test_iterations": int(NUM_TEST_ITERATIONS),
        "prefill_ms": float(prefill_time),
        "all_times_ms": [float(x) for x in all_times_ms],
        "all_wall_times_ms": [float(x) for x in all_wall_times_ms],
    }

    return result, tokens


def test_dense_pruned():
    print("\n" + "=" * 80)
    print("测试: DENSE PRUNED (剪枝后权重 + dense GEMV)")
    print("=" * 80)

    print("\n[准备] 加载剪枝后权重（dense 格式）...")
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            proj_module = get_projection_module(layer, proj_name)
            proj_module.forward = original_forwards[layer_idx][proj_name]
            pruned_weight = sparse_params[layer_idx][proj_name]["pruned_weight"]
            proj_module.weight.data = pruned_weight.T.to(DEVICE)
        layer.self_attn.forward = original_block_forwards[layer_idx]["self_attn"]
        layer.mlp.forward = original_block_forwards[layer_idx]["mlp"]

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        torch.cuda.synchronize()
        start_event.record()
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask_tok,
            use_cache=True,
            return_dict=True,
        )
        end_event.record()
        torch.cuda.synchronize()

    prefill_time = start_event.elapsed_time(end_event)
    past_key_values = outputs.past_key_values
    attention_mask = attention_mask_tok
    del outputs
    torch.cuda.empty_cache()

    last_indices = attention_mask.sum(dim=1) - 1
    cur_input_ids = input_ids.gather(1, last_indices.unsqueeze(1))

    all_times_ms = []
    all_wall_times_ms = []
    tokens = []

    with torch.no_grad():
        for test_iter in range(NUM_TEST_ITERATIONS):
            decode_past = past_key_values
            decode_attn_mask = attention_mask.clone()
            decode_input_ids = cur_input_ids.clone()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start_event.record()

            for _ in range(NUM_DECODE_STEPS):
                new_token_mask = torch.ones((batch_size, 1), device=decode_attn_mask.device, dtype=decode_attn_mask.dtype)
                cur_attn_mask = torch.cat([decode_attn_mask, new_token_mask], dim=1)
                outputs = model(
                    input_ids=decode_input_ids,
                    past_key_values=decode_past,
                    attention_mask=cur_attn_mask,
                    use_cache=True,
                    return_dict=True,
                )
                decode_past = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                if test_iter == 0:
                    tokens.append(next_token.detach())
                decode_input_ids = next_token.unsqueeze(-1)
                decode_attn_mask = cur_attn_mask

            end_event.record()
            torch.cuda.synchronize()
            iter_time_ms = start_event.elapsed_time(end_event)
            iter_wall_ms = (time.perf_counter() - t0) * 1000.0
            all_times_ms.append(iter_time_ms)
            all_wall_times_ms.append(iter_wall_ms)

            if (test_iter + 1) % 10 == 0:
                print(f"  完成 {test_iter + 1}/{NUM_TEST_ITERATIONS} 次测试")

    import numpy as np

    total_ms = np.mean(all_times_ms)
    wall_ms = np.mean(all_wall_times_ms)
    ms_per_token = total_ms / NUM_DECODE_STEPS
    tokens_per_sec = batch_size * 1000.0 / ms_per_token

    result = {
        "mode": "dense_pruned",
        "total_ms": float(total_ms),
        "wall_ms": float(wall_ms),
        "ms_per_token": float(ms_per_token),
        "tokens_per_sec": float(tokens_per_sec),
        "batch_size": int(batch_size),
        "seq_len": int(SEQ_LEN),
        "num_decode_steps": int(NUM_DECODE_STEPS),
        "num_test_iterations": int(NUM_TEST_ITERATIONS),
        "prefill_ms": float(prefill_time),
        "all_times_ms": [float(x) for x in all_times_ms],
        "all_wall_times_ms": [float(x) for x in all_wall_times_ms],
    }

    return result, tokens


def test_dense_original():
    print("\n" + "=" * 80)
    print("测试: DENSE ORIGINAL (原始权重 + dense GEMV)")
    print("=" * 80)

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            proj_module = get_projection_module(layer, proj_name)
            proj_module.forward = original_forwards[layer_idx][proj_name]
            original_weight = sparse_params[layer_idx][proj_name]["original_weight"]
            proj_module.weight.data = original_weight.T.to(DEVICE)
        layer.self_attn.forward = original_block_forwards[layer_idx]["self_attn"]
        layer.mlp.forward = original_block_forwards[layer_idx]["mlp"]

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        torch.cuda.synchronize()
        start_event.record()
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask_tok,
            use_cache=True,
            return_dict=True,
        )
        end_event.record()
        torch.cuda.synchronize()

    prefill_time = start_event.elapsed_time(end_event)
    past_key_values = outputs.past_key_values
    attention_mask = attention_mask_tok
    del outputs
    torch.cuda.empty_cache()

    last_indices = attention_mask.sum(dim=1) - 1
    cur_input_ids = input_ids.gather(1, last_indices.unsqueeze(1))

    all_times_ms = []
    all_wall_times_ms = []
    tokens = []

    with torch.no_grad():
        for test_iter in range(NUM_TEST_ITERATIONS):
            decode_past = past_key_values
            decode_attn_mask = attention_mask.clone()
            decode_input_ids = cur_input_ids.clone()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start_event.record()

            for _ in range(NUM_DECODE_STEPS):
                new_token_mask = torch.ones((batch_size, 1), device=decode_attn_mask.device, dtype=decode_attn_mask.dtype)
                cur_attn_mask = torch.cat([decode_attn_mask, new_token_mask], dim=1)
                outputs = model(
                    input_ids=decode_input_ids,
                    past_key_values=decode_past,
                    attention_mask=cur_attn_mask,
                    use_cache=True,
                    return_dict=True,
                )
                decode_past = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                if test_iter == 0:
                    tokens.append(next_token.detach())
                decode_input_ids = next_token.unsqueeze(-1)
                decode_attn_mask = cur_attn_mask

            end_event.record()
            torch.cuda.synchronize()
            iter_time_ms = start_event.elapsed_time(end_event)
            iter_wall_ms = (time.perf_counter() - t0) * 1000.0
            all_times_ms.append(iter_time_ms)
            all_wall_times_ms.append(iter_wall_ms)

            if (test_iter + 1) % 10 == 0:
                print(f"  完成 {test_iter + 1}/{NUM_TEST_ITERATIONS} 次测试")

    import numpy as np

    total_ms = np.mean(all_times_ms)
    wall_ms = np.mean(all_wall_times_ms)
    ms_per_token = total_ms / NUM_DECODE_STEPS
    tokens_per_sec = batch_size * 1000.0 / ms_per_token

    result = {
        "mode": "dense_original",
        "total_ms": float(total_ms),
        "wall_ms": float(wall_ms),
        "ms_per_token": float(ms_per_token),
        "tokens_per_sec": float(tokens_per_sec),
        "batch_size": int(batch_size),
        "seq_len": int(SEQ_LEN),
        "num_decode_steps": int(NUM_DECODE_STEPS),
        "num_test_iterations": int(NUM_TEST_ITERATIONS),
        "prefill_ms": float(prefill_time),
        "all_times_ms": [float(x) for x in all_times_ms],
        "all_wall_times_ms": [float(x) for x in all_wall_times_ms],
    }

    return result, tokens


if __name__ == "__main__":
    if TEST_MODE == "sparse":
        result, tokens = test_sparse()
    elif TEST_MODE == "dense_pruned":
        result, tokens = test_dense_pruned()
    else:
        result, tokens = test_dense_original()

    output_dir = "/wangqitong/4_8/logs_4_8"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{TEST_MODE}_result.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 80)
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 80)
