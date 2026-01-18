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
from collections import defaultdict

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
import threading

# 添加工具函数路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nmsparse_test_utils import load_and_prune_weight, convert_to_sparse_format

# ===== 配置 =====
LLAMA_MODEL_PATH = "/wangqitong/llama_2-7b"
DEVICE = "cuda"
DTYPE = torch.float32

PROMPT_TEXT = "The future of artificial intelligence is"
NUM_DECODE_STEPS = int(os.getenv('E2E_STEPS', '1'))
NUM_WARMUP_DECODE = 0
NUM_TEST_ITERATIONS = int(os.getenv('E2E_ITERS', '100'))  # 每个模式测试的次数
BATCH_SIZE = int(os.getenv('E2E_BATCH', '8'))
SEQ_LEN = int(os.getenv('E2E_SEQ_LEN', '512'))

OOM_SWEEP = os.getenv("E2E_OOM_SWEEP", "0") == "1"
SWEEP_REPEATS = int(os.getenv("E2E_SWEEP_REPEATS", "20"))
SWEEP_MAX_STEPS = int(os.getenv("E2E_SWEEP_MAX_STEPS", "4096"))
SWEEP_STEP = int(os.getenv("E2E_SWEEP_STEP", "1"))

SWEEP_KVLEN_START = int(os.getenv("E2E_SWEEP_KVLEN_START", "0"))
SWEEP_KVLEN_END = int(os.getenv("E2E_SWEEP_KVLEN_END", "0"))
SWEEP_KVLEN_STEP = int(os.getenv("E2E_SWEEP_KVLEN_STEP", "0"))

# 稀疏化配置 - 2:4 稀疏模式
SPARSITY = 0.5
NUM_BANK_VAL = 4  # 修改为4，实现2:4稀疏（从32改为4）
NUM_THREADS = 128
PROFILE_COMPONENTS = False  # 是否测量组件时间

# 获取测试模式
TEST_MODE = os.getenv('E2E_MODE', 'sparse')
if TEST_MODE not in ['sparse', 'dense_pruned', 'dense_original']:
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
print(f"  稀疏化范围: 所有 32 层的 QKVO + MLP (7个投影/层, 2:4稀疏模式)")
print(f"  Profile components: {PROFILE_COMPONENTS}")
print("=" * 80)

# ===== 禁用 TF32 =====
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

# ===== 加载 nmSPARSE kernel =====
print("\n加载 nmSPARSE kernel...")
script_dir = os.path.dirname(os.path.abspath(__file__))
wrapper_cu_path = os.path.join(script_dir, "nmsparse_wrapper_with_initialdata.cu")

if not os.path.exists(wrapper_cu_path):
    print(f"✗ 找不到 CUDA wrapper: {wrapper_cu_path}")
    sys.exit(1)

try:
    import os
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    cuda_include_path = f"{conda_prefix}/targets/x86_64-linux/include"
    
    nmsparse_module = load(
        name='nmsparse_fig9_2_4',
        sources=[wrapper_cu_path],
        extra_cuda_cflags=[
            '-O3', '--use_fast_math',
            f'-I{cuda_include_path}'
        ],
        verbose=True
    )
    print("✓ nmSPARSE kernel 编译/加载成功")
except Exception as e:
    print(f"✗ Kernel 编译/加载失败: {e}")
    sys.exit(1)

# ===== 加载模型和 tokenizer =====
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

# ===== 加载或生成剪枝权重 =====
pruned_cache_file = f"/wangqitong/sparse_params_sparsity0.5_bank4_mlp_qkvo_2_4.pt"

if os.path.exists(pruned_cache_file):
    print(f"\n从缓存加载剪枝参数: {pruned_cache_file}")
    sparse_params = torch.load(pruned_cache_file, map_location='cpu')
    print(f"✓ 剪枝参数加载完成")
    for layer_idx in range(num_layers):
        for proj_name in sparse_params[layer_idx].keys():
            entry = sparse_params[layer_idx][proj_name]
            K = entry['K']
            N = entry['N']
            num_bank = K // NUM_BANK_VAL
            num_nonzeros_per_bank = int(NUM_BANK_VAL * (1 - SPARSITY))
            entry['w'] = num_bank * num_nonzeros_per_bank
            entry['h'] = N
            entry['block_width'] = num_nonzeros_per_bank
            entry['vec_width'] = NUM_BANK_VAL
else:
    print(f"\n缓存不存在，开始剪枝所有 32 层的 QKVO + MLP...")
    _prune_dev = next(model.parameters()).device
    print(f"  [Pruning] load_and_prune_weight 将在设备上执行: {_prune_dev} ({'GPU' if _prune_dev.type == 'cuda' else 'CPU'})")
    sparse_params = {}
    for layer_idx in range(num_layers):
        sparse_params[layer_idx] = {}
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            real_weight, pruned_weight, pruned_mask, K, N = load_and_prune_weight(
                model, layer_idx, proj_name, SPARSITY, NUM_BANK_VAL
            )
            mat_data_cpu, mat_index_cpu = convert_to_sparse_format(
                pruned_weight, pruned_mask, NUM_BANK_VAL
            )
            
            # 计算稀疏格式的参数
            num_bank = K // NUM_BANK_VAL
            num_nonzeros_per_bank = int(NUM_BANK_VAL * (1 - SPARSITY))
            w = num_bank * num_nonzeros_per_bank
            h = N
            block_width = num_nonzeros_per_bank
            vec_width = NUM_BANK_VAL
            sparse_params[layer_idx][proj_name] = {
                'mat_data': mat_data_cpu,
                'mat_index': mat_index_cpu,
                'pruned_weight': pruned_weight,
                'original_weight': real_weight,
                'w': w, 'h': h,
                'block_width': block_width,
                'vec_width': vec_width,
                'K': K, 'N': N,
            }
        if (layer_idx + 1) % 2 == 0:
            print(f"  ✓ Layer {layer_idx} 剪枝完成 (4 QKVO + 3 MLP)")
    
    os.makedirs(os.path.dirname(pruned_cache_file), exist_ok=True)
    torch.save(sparse_params, pruned_cache_file)
    print(f"✓ 剪枝完成并保存到 {pruned_cache_file}")
    print(f"   总共剪枝: {num_layers} 层 × 7 个投影 = {num_layers * 7} 个权重矩阵")

# 将稀疏参数移到 GPU
print("\n将稀疏参数移到 GPU...")
for layer_idx in range(num_layers):
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        sparse_params[layer_idx][proj_name]['mat_data'] = sparse_params[layer_idx][proj_name]['mat_data'].to(DEVICE)
        sparse_params[layer_idx][proj_name]['mat_index'] = sparse_params[layer_idx][proj_name]['mat_index'].to(DEVICE)
print("✓ 稀疏参数已移到 GPU")


# ===== 定义辅助类 =====
class SparsifiedLinear:
    """稀疏化的线性层包装器"""
    def __init__(self, original_module, nmsparse_module, sparse_param, name, timer_events=None):
        self.original_module = original_module
        self.nmsparse_module = nmsparse_module
        self.name = name
        self._timer_events = timer_events
        
        self.mat_data = sparse_param['mat_data']
        self.mat_index = sparse_param['mat_index']
        self.w = sparse_param['w']
        self.h = sparse_param['h']
        self.block_width = sparse_param['block_width']
        self.vec_width = sparse_param['vec_width']
        self.bias = original_module.bias
    
    def __call__(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_2d = x.reshape(-1, hidden_dim)  # [B*L, K]
        minibatch = x_2d.shape[0]         # B*L
        vecNum = x_2d.shape[1]            # K (hidden_dim)
        
        _se, _ee = None, None
        if self._timer_events is not None:
            _se = torch.cuda.Event(enable_timing=True)
            _ee = torch.cuda.Event(enable_timing=True)
            _se.record()
        
        nvtx.range_push("CPP::nmsparse_forward")
        output = self.nmsparse_module.forward(
            x_2d, self.mat_data, self.mat_index,
            self.w, self.h, self.block_width,
            NUM_THREADS, self.vec_width, minibatch, vecNum
        )
        nvtx.range_pop()  # CPP::nmsparse_forward
        
        if self._timer_events is not None:
            _ee.record()
            self._timer_events.append((_se, _ee))
        
        if self.bias is not None:
            output = output + self.bias
        
        output = output.contiguous().view(batch_size, seq_len, -1)
        return output

class _TimedForward:
    """计时包装器（简化版，已移除细粒度NVTX hooks）"""
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

class _NVTXOnlyForward:
    def __init__(self, original_forward, tag: str):
        self.original_forward = original_forward
        self.tag = tag
    def __call__(self, *args, **kwargs):
        out = self.original_forward(*args, **kwargs)
        return out

# NVTX hooks 已禁用 - 只保留Python→C++→CUDA调用链的关键标记
_ATTN_CTX = threading.local()

# def install_attn_nvtx_hooks():
#     """细粒度NVTX hooks已禁用，只保留主要调用链标记"""
#     pass

# # 安装 NVTX hooks
# install_attn_nvtx_hooks()

# 保存原始 forward 方法
print("\n保存原始 forward 方法...")
original_forwards = {}
original_block_forwards = {}
for layer_idx in range(num_layers):
    original_forwards[layer_idx] = {}
    layer = model.model.layers[layer_idx]
    original_block_forwards[layer_idx] = {
        'self_attn': layer.self_attn.forward,
        'mlp': layer.mlp.forward,
    }
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        proj_module = getattr(layer.self_attn, proj_name)
        original_forwards[layer_idx][proj_name] = proj_module.forward
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj_module = getattr(layer.mlp, proj_name)
        original_forwards[layer_idx][proj_name] = proj_module.forward
print("✓ 原始 forward 已保存")

def get_projection_module(layer, proj_name):
    """根据投影名称获取对应的模块"""
    if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        return getattr(layer.self_attn, proj_name)
    elif proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        return getattr(layer.mlp, proj_name)
    else:
        raise ValueError(f"未知的投影名称: {proj_name}")

# ===== Tokenize =====
print("\nTokenize 输入...")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if getattr(model.config, 'pad_token_id', None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

inputs = tokenizer(
    [PROMPT_TEXT] * BATCH_SIZE,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=SEQ_LEN
)
input_ids = inputs["input_ids"].to(DEVICE)
attention_mask_tok = inputs["attention_mask"].to(DEVICE)
batch_size = input_ids.shape[0]
print(f"✓ Tokenize 完成: batch={batch_size}, seq={input_ids.shape[1]}")


# ==================== 测试函数 ====================


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
    """Sparse 配置测试"""
    print("\n" + "=" * 80)
    print("测试: SPARSE (剪枝后权重 + nmSPARSE kernel)")
    print("=" * 80)
    
    # 事件收集
    qkvo_linear_events = []
    mlp_linear_events = []
    attn_events = []
    mlp_events = []
    
    # Prefill: 使用 dense 格式的剪枝后权重
    print("\n[Prefill] 加载剪枝后权重（dense 格式）...")
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            proj_module = get_projection_module(layer, proj_name)
            pruned_weight = sparse_params[layer_idx][proj_name]['pruned_weight']
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
    
    # 应用稀疏化
    print("\n[Decode] 应用稀疏化...")
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            proj_module = get_projection_module(layer, proj_name)
            _events = None
            if PROFILE_COMPONENTS:
                if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    _events = qkvo_linear_events
                else:
                    _events = mlp_linear_events
            sparse_linear = SparsifiedLinear(
                proj_module, nmsparse_module,
                sparse_params[layer_idx][proj_name],
                f"layer{layer_idx}.{proj_name}",
                _events
            )
            proj_module.forward = sparse_linear
        
        if PROFILE_COMPONENTS:
            layer.self_attn.forward = _TimedForward(
                original_block_forwards[layer_idx]['self_attn'], attn_events, "PY::self_attn"
            )
            layer.mlp.forward = _TimedForward(
                original_block_forwards[layer_idx]['mlp'], mlp_events, "PY::mlp"
            )
    print("✓ 稀疏化已应用")
    print(f"  - nmSPARSE 2:4: 7 个投影 × 32 层 = 224 个矩阵 (包含 QKVO + MLP)")

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
    
    # Decode - 测试 100 次
    print(f"\n[Decode] 运行 {NUM_TEST_ITERATIONS} 次测试（每次 {NUM_DECODE_STEPS} steps）...")
    all_times_ms = []
    all_wall_times_ms = []
    tokens = []
    kvlen_before = int(attention_mask_tok.size(1))
    
    with torch.no_grad():
        for test_iter in range(NUM_TEST_ITERATIONS):
            nvtx.range_push(f"decode_iter::{test_iter}")
            try:
                # 重置状态
                decode_past = past_key_values
                decode_attn_mask = attention_mask.clone()
                decode_input_ids = cur_input_ids.clone()
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                start_event.record()
                
                for step in range(NUM_DECODE_STEPS):
                    new_token_mask = torch.ones(
                        (batch_size, 1),
                        device=decode_attn_mask.device,
                        dtype=decode_attn_mask.dtype,
                    )
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
                    if test_iter == 0:  # 只保存第一次的 tokens
                        tokens.append(next_token.detach())
                    next_token = next_token.unsqueeze(-1)
                    
                    decode_attn_mask = cur_attn_mask
                    decode_input_ids = next_token
                
                end_event.record()
                torch.cuda.synchronize()
                iter_time_ms = start_event.elapsed_time(end_event)
                iter_wall_ms = (time.perf_counter() - t0) * 1000.0
                
                all_times_ms.append(iter_time_ms)
                all_wall_times_ms.append(iter_wall_ms)
                
                if (test_iter + 1) % 10 == 0:
                    print(f"  完成 {test_iter + 1}/{NUM_TEST_ITERATIONS} 次测试")
            finally:
                nvtx.range_pop()  # decode_iter
    
    # 计算统计指标
    import numpy as np
    total_ms = np.mean(all_times_ms)
    wall_ms = np.mean(all_wall_times_ms)
    ms_per_token = total_ms / NUM_DECODE_STEPS
    tokens_per_sec = batch_size * 1000.0 / ms_per_token
    
    # 组件时间（需要除以测试次数得到平均值）
    qkvo_ms, attn_core_ms, mlp_ms = 0.0, 0.0, 0.0
    mlp_linear_ms, mlp_core_ms = 0.0, 0.0
    if PROFILE_COMPONENTS:
        torch.cuda.synchronize()
        qkvo_ms = sum(se.elapsed_time(ee) for se, ee in qkvo_linear_events) / NUM_TEST_ITERATIONS
        mlp_linear_ms = sum(se.elapsed_time(ee) for se, ee in mlp_linear_events) / NUM_TEST_ITERATIONS
        attn_ms = sum(se.elapsed_time(ee) for se, ee in attn_events) / NUM_TEST_ITERATIONS
        mlp_ms = sum(se.elapsed_time(ee) for se, ee in mlp_events) / NUM_TEST_ITERATIONS
        attn_core_ms = max(0.0, attn_ms - qkvo_ms)
        mlp_core_ms = max(0.0, mlp_ms - mlp_linear_ms)
    
    # 打印结果
    print(f"\n{'=' * 80}")
    print("✓ Sparse Decode 完成")
    print(f"\n【所有 {NUM_TEST_ITERATIONS} 次测试时间（GPU Event）】")
    for i, t in enumerate(all_times_ms, 1):
        print(f"  第 {i:3d} 次: {t:.3f} ms")
    
    print(f"\n【统计结果】")
    print(f"  平均时间（GPU）: {total_ms:.3f} ms")
    print(f"  平均时间（Wall）: {wall_ms:.3f} ms")
    print(f"  最小时间: {np.min(all_times_ms):.3f} ms")
    print(f"  最大时间: {np.max(all_times_ms):.3f} ms")
    print(f"  标准差: {np.std(all_times_ms):.3f} ms")
    print(f"  ms/token: {ms_per_token:.3f}")
    print(f"  tokens/s: {tokens_per_sec:.1f}")
    print(f"  Decode-1step (B={batch_size}, KVlen={kvlen_before}): "
          f"GPU {total_ms/NUM_DECODE_STEPS:.3f} ms, Wall {wall_ms/NUM_DECODE_STEPS:.3f} ms")
    
    if PROFILE_COMPONENTS:
        print(
            f"  [占比] QKVO-linear: {qkvo_ms/total_ms*100:.1f}%, "
            f"Attention(core): {attn_core_ms/total_ms*100:.1f}%, "
            f"MLP-linear: {mlp_linear_ms/total_ms*100:.1f}%, "
            f"MLP(core): {mlp_core_ms/total_ms*100:.1f}%, "
            f"其他: {max(0.0, 100-(qkvo_ms+attn_core_ms+mlp_linear_ms+mlp_core_ms)/total_ms*100):.1f}%"
        )
    
    # 保存结果
    result = {
        'mode': 'sparse',
        'total_ms': total_ms,
        'wall_ms': wall_ms,
        'ms_per_token': ms_per_token,
        'tokens_per_sec': tokens_per_sec,
        'batch_size': batch_size,
        'kvlen': kvlen_before,
        'num_decode_steps': NUM_DECODE_STEPS,
        'num_test_iterations': NUM_TEST_ITERATIONS,
        'prefill_ms': prefill_time,
        'all_times_ms': all_times_ms,
        'all_wall_times_ms': all_wall_times_ms,
        'min_ms': float(np.min(all_times_ms)),
        'max_ms': float(np.max(all_times_ms)),
        'std_ms': float(np.std(all_times_ms)),
    }
    if PROFILE_COMPONENTS:
        result.update({
            'qkvo_ms': qkvo_ms,
            'attn_core_ms': attn_core_ms,
            'mlp_ms': mlp_ms,
            'mlp_linear_ms': mlp_linear_ms,
            'mlp_core_ms': mlp_core_ms,
        })
    
    return result, tokens


def test_dense_pruned():
    """Dense (pruned) 配置测试 - 与 test_sparse 类似但不使用稀疏 kernel"""
    print("\n" + "=" * 80)
    print("测试: DENSE PRUNED (剪枝后权重 + dense GEMV)")
    print("=" * 80)
    
    qkvo_events, attn_events, mlp_events = [], [], []
    
    # 恢复原始 forward，加载剪枝后权重
    print("\n[准备] 加载剪枝后权重（dense 格式）...")
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            proj_module = get_projection_module(layer, proj_name)
            proj_module.forward = original_forwards[layer_idx][proj_name]
            pruned_weight = sparse_params[layer_idx][proj_name]['pruned_weight']
            proj_module.weight.data = pruned_weight.T.to(DEVICE)
        layer.self_attn.forward = original_block_forwards[layer_idx]['self_attn']
        layer.mlp.forward = original_block_forwards[layer_idx]['mlp']
    print("✓ 剪枝后权重已加载（dense 格式）")
    
    # Prefill
    print("\n[Prefill] 生成 KV cache...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        torch.cuda.synchronize()
        start_event.record()
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask_tok,
                            use_cache=True, return_dict=True)
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
    
    # Decode - 测试 100 次
    print(f"\n[Decode] 运行 {NUM_TEST_ITERATIONS} 次测试（每次 {NUM_DECODE_STEPS} steps）...")
    all_times_ms = []
    all_wall_times_ms = []
    tokens = []
    kvlen_before = int(attention_mask_tok.size(1))
    
    with torch.no_grad():
        for test_iter in range(NUM_TEST_ITERATIONS):
            nvtx.range_push(f"decode_iter::{test_iter}")
            try:
                decode_past = past_key_values
                decode_attn_mask = attention_mask.clone()
                decode_input_ids = cur_input_ids.clone()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                start_event.record()
                for step in range(NUM_DECODE_STEPS):
                    new_token_mask = torch.ones((batch_size, 1), device=decode_attn_mask.device, dtype=decode_attn_mask.dtype)
                    cur_attn_mask = torch.cat([decode_attn_mask, new_token_mask], dim=1)
                    outputs = model(input_ids=decode_input_ids, past_key_values=decode_past,
                                  attention_mask=cur_attn_mask, use_cache=True, return_dict=True)
                    decode_past = outputs.past_key_values
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                    if test_iter == 0:
                        tokens.append(next_token.detach())
                    next_token = next_token.unsqueeze(-1)
                    decode_attn_mask = cur_attn_mask
                    decode_input_ids = next_token
                end_event.record()
                torch.cuda.synchronize()
                iter_time_ms = start_event.elapsed_time(end_event)
                iter_wall_ms = (time.perf_counter() - t0) * 1000.0
                all_times_ms.append(iter_time_ms)
                all_wall_times_ms.append(iter_wall_ms)
                if (test_iter + 1) % 10 == 0:
                    print(f"  完成 {test_iter + 1}/{NUM_TEST_ITERATIONS} 次测试")
            finally:
                nvtx.range_pop()  # decode_iter
    
    import numpy as np
    total_ms = np.mean(all_times_ms)
    wall_ms = np.mean(all_wall_times_ms)
    ms_per_token = total_ms / NUM_DECODE_STEPS
    tokens_per_sec = batch_size * 1000.0 / ms_per_token
    
    print(f"\n{'=' * 80}")
    print("✓ Dense (pruned) Decode 完成")
    print(f"\n【所有 {NUM_TEST_ITERATIONS} 次测试时间（GPU Event）】")
    for i, t in enumerate(all_times_ms, 1):
        print(f"  第 {i:3d} 次: {t:.3f} ms")
    print(f"\n【统计结果】")
    print(f"  平均时间（GPU）: {total_ms:.3f} ms")
    print(f"  平均时间（Wall）: {wall_ms:.3f} ms")
    print(f"  最小时间: {np.min(all_times_ms):.3f} ms")
    print(f"  最大时间: {np.max(all_times_ms):.3f} ms")
    print(f"  标准差: {np.std(all_times_ms):.3f} ms")
    print(f"  ms/token: {ms_per_token:.3f}")
    print(f"  tokens/s: {tokens_per_sec:.1f}")
    print(f"  Decode-1step (B={batch_size}, KVlen={kvlen_before}): GPU {total_ms/NUM_DECODE_STEPS:.3f} ms, Wall {wall_ms/NUM_DECODE_STEPS:.3f} ms")
    
    result = {
        'mode': 'dense_pruned',
        'total_ms': total_ms,
        'wall_ms': wall_ms,
        'ms_per_token': ms_per_token,
        'tokens_per_sec': tokens_per_sec,
        'batch_size': batch_size,
        'kvlen': kvlen_before,
        'num_decode_steps': NUM_DECODE_STEPS,
        'num_test_iterations': NUM_TEST_ITERATIONS,
        'prefill_ms': prefill_time,
        'all_times_ms': all_times_ms,
        'all_wall_times_ms': all_wall_times_ms,
        'min_ms': float(np.min(all_times_ms)),
        'max_ms': float(np.max(all_times_ms)),
        'std_ms': float(np.std(all_times_ms)),
    }
    return result, tokens


def test_dense_original():
    """Dense (original) 配置测试 - 使用原始未剪枝权重"""
    print("\n" + "=" * 80)
    print("测试: DENSE ORIGINAL (原始权重 + dense GEMV)")
    print("=" * 80)
    
    # 恢复原始权重
    print("\n[准备] 加载原始权重...")
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            proj_module = get_projection_module(layer, proj_name)
            proj_module.forward = original_forwards[layer_idx][proj_name]
            original_weight = sparse_params[layer_idx][proj_name]['original_weight']
            proj_module.weight.data = original_weight.T.to(DEVICE)
        layer.self_attn.forward = original_block_forwards[layer_idx]['self_attn']
        layer.mlp.forward = original_block_forwards[layer_idx]['mlp']
    print("✓ 原始权重已加载")
    
    # Prefill
    print("\n[Prefill] 生成 KV cache...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        torch.cuda.synchronize()
        start_event.record()
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask_tok,
                            use_cache=True, return_dict=True)
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
    
    # Decode - 测试 100 次
    print(f"\n[Decode] 运行 {NUM_TEST_ITERATIONS} 次测试（每次 {NUM_DECODE_STEPS} steps）...")
    all_times_ms = []
    all_wall_times_ms = []
    tokens = []
    kvlen_before = int(attention_mask_tok.size(1))
    
    with torch.no_grad():
        for test_iter in range(NUM_TEST_ITERATIONS):
            nvtx.range_push(f"decode_iter::{test_iter}")
            try:
                decode_past = past_key_values
                decode_attn_mask = attention_mask.clone()
                decode_input_ids = cur_input_ids.clone()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                start_event.record()
                for step in range(NUM_DECODE_STEPS):
                    new_token_mask = torch.ones((batch_size, 1), device=decode_attn_mask.device, dtype=decode_attn_mask.dtype)
                    cur_attn_mask = torch.cat([decode_attn_mask, new_token_mask], dim=1)
                    outputs = model(input_ids=decode_input_ids, past_key_values=decode_past,
                                  attention_mask=cur_attn_mask, use_cache=True, return_dict=True)
                    decode_past = outputs.past_key_values
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                    if test_iter == 0:
                        tokens.append(next_token.detach())
                    next_token = next_token.unsqueeze(-1)
                    decode_attn_mask = cur_attn_mask
                    decode_input_ids = next_token
                end_event.record()
                torch.cuda.synchronize()
                iter_time_ms = start_event.elapsed_time(end_event)
                iter_wall_ms = (time.perf_counter() - t0) * 1000.0
                all_times_ms.append(iter_time_ms)
                all_wall_times_ms.append(iter_wall_ms)
                if (test_iter + 1) % 10 == 0:
                    print(f"  完成 {test_iter + 1}/{NUM_TEST_ITERATIONS} 次测试")
            finally:
                nvtx.range_pop()  # decode_iter
    
    import numpy as np
    total_ms = np.mean(all_times_ms)
    wall_ms = np.mean(all_wall_times_ms)
    ms_per_token = total_ms / NUM_DECODE_STEPS
    tokens_per_sec = batch_size * 1000.0 / ms_per_token
    
    print(f"\n{'=' * 80}")
    print("✓ Dense (original) Decode 完成")
    print(f"\n【所有 {NUM_TEST_ITERATIONS} 次测试时间（GPU Event）】")
    for i, t in enumerate(all_times_ms, 1):
        print(f"  第 {i:3d} 次: {t:.3f} ms")
    print(f"\n【统计结果】")
    print(f"  平均时间（GPU）: {total_ms:.3f} ms")
    print(f"  平均时间（Wall）: {wall_ms:.3f} ms")
    print(f"  最小时间: {np.min(all_times_ms):.3f} ms")
    print(f"  最大时间: {np.max(all_times_ms):.3f} ms")
    print(f"  标准差: {np.std(all_times_ms):.3f} ms")
    print(f"  ms/token: {ms_per_token:.3f}")
    print(f"  tokens/s: {tokens_per_sec:.1f}")
    print(f"  Decode-1step (B={batch_size}, KVlen={kvlen_before}): GPU {total_ms/NUM_DECODE_STEPS:.3f} ms, Wall {wall_ms/NUM_DECODE_STEPS:.3f} ms")
    
    result = {
        'mode': 'dense_original',
        'total_ms': total_ms,
        'wall_ms': wall_ms,
        'ms_per_token': ms_per_token,
        'tokens_per_sec': tokens_per_sec,
        'batch_size': batch_size,
        'kvlen': kvlen_before,
        'num_decode_steps': NUM_DECODE_STEPS,
        'num_test_iterations': NUM_TEST_ITERATIONS,
        'prefill_ms': prefill_time,
        'all_times_ms': all_times_ms,
        'all_wall_times_ms': all_wall_times_ms,
        'min_ms': float(np.min(all_times_ms)),
        'max_ms': float(np.max(all_times_ms)),
        'std_ms': float(np.std(all_times_ms)),
    }
    return result, tokens


# ==================== 主程序 ====================

if __name__ == '__main__':
    # 根据模式运行对应测试
    if TEST_MODE == 'sparse':
        result, tokens = test_sparse()
    elif TEST_MODE == 'dense_pruned':
        result, tokens = test_dense_pruned()
    elif TEST_MODE == 'dense_original':
        result, tokens = test_dense_original()
    
    # 保存结果到文件
    output_dir = "/wangqitong/2_4/logs_mlp_2_4"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{TEST_MODE}_result.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 80)
