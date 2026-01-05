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
import threading
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer
from torch.utils.cpp_extension import load
import torch.cuda.nvtx as nvtx

# 添加工具函数路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nmsparse_test_utils import load_and_prune_weight, convert_to_sparse_format

# ===== 配置 =====
LLAMA_MODEL_PATH = "/root/autodl-tmp/llama_2-7b"
DEVICE = "cuda"
DTYPE = torch.float32

PROMPT_TEXT = "The future of artificial intelligence is"
NUM_DECODE_STEPS = 1
NUM_WARMUP_DECODE = 0
NUM_TEST_ITERATIONS = 100  # 每个模式测试的次数
BATCH_SIZE = 1
SEQ_LEN = 512

# 稀疏化配置
SPARSITY = 0.5
NUM_BANK_VAL = 32
NUM_THREADS = 128
PROFILE_COMPONENTS = False  # 是否测量组件时间（禁用以避免CUDA Event overhead）

# 获取测试模式
TEST_MODE = os.getenv('E2E_MODE', 'dense_original')
print(f"\n测试模式: {TEST_MODE}")
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
print(f"  稀疏化范围: 所有 32 层的 QKVO + MLP (7个投影/层)")
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
        name='nmsparse_fig9',
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

# NVTX hooks 已禁用 - 只保留Python→C++→CUDA调用链的关键标记
_ATTN_CTX = threading.local()

# def install_attn_nvtx_hooks():
#     """细粒度NVTX hooks已禁用，只保留主要调用链标记"""
#     pass

# # 安装 NVTX hooks
# install_attn_nvtx_hooks()

# ===== 加载或生成剪枝权重 =====
pruned_cache_file = f"/root/autodl-tmp/sparse_params_sparsity0.5_bank32_mlp_qkvo.pt"

if os.path.exists(pruned_cache_file):
    print(f"\n从缓存加载剪枝参数: {pruned_cache_file}")
    sparse_params = torch.load(pruned_cache_file, map_location='cpu')
    print(f"✓ 剪枝参数加载完成")
    # 兼容旧缓存：重新计算派生的稀疏布局参数，确保与 kernel 假设一致
    #   vec_width = NUM_BANK_VAL
    #   block_width = int(NUM_BANK_VAL * (1 - SPARSITY))
    #   w = block_width * num_bank
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
    sparse_params = {}
    for layer_idx in range(num_layers):
        sparse_params[layer_idx] = {}
        # for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            real_weight, pruned_weight, pruned_mask, K, N = load_and_prune_weight(
                model, layer_idx, proj_name, SPARSITY
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
        if (layer_idx + 1) % 8 == 0:
            print(f"  ✓ Layer {layer_idx} 剪枝完成 (仅 4 个 QKVO 投影)")
    
    os.makedirs(os.path.dirname(pruned_cache_file), exist_ok=True)
    torch.save(sparse_params, pruned_cache_file)
    print(f"✓ 剪枝完成并保存到 {pruned_cache_file}")
    print(f"   总共剪枝: {num_layers} 层 × 4 个投影 = {num_layers * 4} 个权重矩阵 (仅 QKVO)")

# 将稀疏参数移到 GPU
print("\n将稀疏参数移到 GPU...")
for layer_idx in range(num_layers):
    layer = model.model.layers[layer_idx]
    # for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        proj_entry = sparse_params[layer_idx][proj_name]

        if 'original_weight' not in proj_entry:
            if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj_module = getattr(layer.self_attn, proj_name)
            else:
                proj_module = getattr(layer.mlp, proj_name)

            proj_entry['original_weight'] = proj_module.weight.data.detach().to('cpu').t().contiguous()

        sparse_params[layer_idx][proj_name]['mat_data'] = sparse_params[layer_idx][proj_name]['mat_data'].to(DEVICE)
        sparse_params[layer_idx][proj_name]['mat_index'] = sparse_params[layer_idx][proj_name]['mat_index'].to(DEVICE)
        sparse_params[layer_idx][proj_name]['pruned_weight'] = sparse_params[layer_idx][proj_name]['pruned_weight'].to(DEVICE)
        # sparse_params[layer_idx][proj_name]['original_weight'] = sparse_params[layer_idx][proj_name]['original_weight'].to(DEVICE)
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
        nvtx.range_push(f"PY::nmsparse_linear::{self.name}")
        
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
        )# nmsparse kernel
        nvtx.range_pop()  # CPP::nmsparse_forward
        
        if self._timer_events is not None:
            _ee.record()
            self._timer_events.append((_se, _ee))
        
        if self.bias is not None:
            output = output + self.bias
        
        output = output.contiguous().view(batch_size, seq_len, -1)
        
        nvtx.range_pop()  # PY::nmsparse_linear
        return output

class _TimedForward:
    """计时包装器（简化版，已移除细粒度NVTX hooks）"""
    def __init__(self, original_forward, events_list, tag="PY::dense_linear"):
        self.original_forward = original_forward
        self.events_list = events_list
        self.tag = tag
    
    def __call__(self, *args, **kwargs):
        nvtx.range_push(self.tag)
        
        se, ee = None, None
        if self.events_list is not None:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
        result = self.original_forward(*args, **kwargs)
        if self.events_list is not None:
            ee.record()
            self.events_list.append((se, ee))
        
        nvtx.range_pop()
        return result

class _NVTXOnlyForward:
    def __init__(self, original_forward, tag: str):
        self.original_forward = original_forward
        self.tag = tag
    def __call__(self, *args, **kwargs):
        nvtx.range_push(self.tag)
        out = self.original_forward(*args, **kwargs)
        nvtx.range_pop()
        return out

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

def test_sparse():
    """Sparse 配置测试"""
    print("\n" + "=" * 80)
    print("测试: SPARSE (剪枝后权重 + nmSPARSE kernel)")
    print("=" * 80)
    
    # 事件收集
    qkvo_events = []
    attn_events = []
    mlp_events = []
    
    # Prefill: 使用已加载的剪枝后权重（在显存优化阶段已加载）
    print("\n[Prefill] 使用已加载的剪枝后权重进行 prefill...")
    # 注意：剪枝后权重已在显存优化阶段加载到模型中，无需重复加载
    
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
        # 只对QKVO投影应用稀疏化，MLP保持密集计算
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:  # 'gate_proj', 'up_proj', 'down_proj']:
            proj_module = get_projection_module(layer, proj_name)
            sparse_linear = SparsifiedLinear(
                proj_module, nmsparse_module,
                sparse_params[layer_idx][proj_name],
                f"layer{layer_idx}.{proj_name}",
                qkvo_events if PROFILE_COMPONENTS else None
            )
            proj_module.forward = sparse_linear   # 替换为稀疏化版本
        
        if PROFILE_COMPONENTS:
            layer.self_attn.forward = _TimedForward(
                original_block_forwards[layer_idx]['self_attn'], attn_events, "PY::self_attn"
            )
            layer.mlp.forward = _TimedForward(
                original_block_forwards[layer_idx]['mlp'], mlp_events, "PY::mlp"
            )
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj_module = getattr(layer.mlp, proj_name)
                proj_module.forward = _TimedForward(
                    original_forwards[layer_idx][proj_name], None, f"PY::mlp::{proj_name}"
                )
    print("✓ 稀疏化已应用")
    print(f"  - nmSPARSE: 7 个投影 × 32 层 = 224 个矩阵 (QKVO + MLP 全部稀疏)")
    print(f"  - 所有线性层均使用 nmSPARSE 计算")
    
    # Decode - 测试 100 次
    print(f"\n[Decode] 运行 {NUM_TEST_ITERATIONS} 次测试（每次 {NUM_DECODE_STEPS} steps）...")
    all_times_ms = []
    all_wall_times_ms = []
    tokens = []
    kvlen_before = int(attention_mask_tok.size(1))
    
    # 使用 NVTX 捕获范围 + 可选的 torch profiler（可通过环境变量关闭）
    nvtx.range_push("CAPTURE")
    if os.getenv('E2E_DISABLE_TORCH_PROFILER', '0') != '1':
        torch.cuda.profiler.start()
    
    with torch.no_grad():
        for test_iter in range(NUM_TEST_ITERATIONS):
            # 重置状态
            decode_past = past_key_values
            decode_attn_mask = attention_mask.clone()
            decode_input_ids = cur_input_ids.clone()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            # NVTX范围标记整个迭代（用于capture-range）
            nvtx.range_push(f"DECODE_ITER_{test_iter}")
            start_event.record()
            
            for step in range(NUM_DECODE_STEPS):
                new_token_mask = torch.ones(
                    (batch_size, 1),
                    device=decode_attn_mask.device,
                    dtype=decode_attn_mask.dtype,
                )
                
                cur_attn_mask = torch.cat([decode_attn_mask, new_token_mask], dim=1)
                
                nvtx.range_push("PY::model_forward")
                outputs = model(
                    input_ids=decode_input_ids,
                    past_key_values=decode_past,
                    attention_mask=cur_attn_mask,
                    use_cache=True,
                    return_dict=True,
                )
                nvtx.range_pop()  # PY::model_forward
                
                decode_past = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                if test_iter == 0:  # 只保存第一次的 tokens
                    tokens.append(next_token.detach())
                next_token = next_token.unsqueeze(-1)
                
                decode_attn_mask = cur_attn_mask
                decode_input_ids = next_token
            
            end_event.record()
            nvtx.range_pop()  # DECODE_ITER_{test_iter}
            
            torch.cuda.synchronize()
            iter_time_ms = start_event.elapsed_time(end_event)
            iter_wall_ms = (time.perf_counter() - t0) * 1000.0
            
            all_times_ms.append(iter_time_ms)
            all_wall_times_ms.append(iter_wall_ms)
            
            if (test_iter + 1) % 10 == 0:
                print(f"  完成 {test_iter + 1}/{NUM_TEST_ITERATIONS} 次测试")
    
    # 停止 profiler / 关闭 CAPTURE 范围
    if os.getenv('E2E_DISABLE_TORCH_PROFILER', '0') != '1':
        torch.cuda.profiler.stop()
    nvtx.range_pop()
    
    # 计算统计指标
    import numpy as np
    total_ms = np.mean(all_times_ms)
    wall_ms = np.mean(all_wall_times_ms)
    ms_per_token = total_ms / NUM_DECODE_STEPS
    tokens_per_sec = batch_size * 1000.0 / ms_per_token
    
    # 组件时间（需要除以测试次数得到平均值）
    qkvo_ms, attn_core_ms, mlp_ms = 0.0, 0.0, 0.0
    if PROFILE_COMPONENTS:
        torch.cuda.synchronize()
        qkvo_ms = sum(se.elapsed_time(ee) for se, ee in qkvo_events) / NUM_TEST_ITERATIONS
        attn_ms = sum(se.elapsed_time(ee) for se, ee in attn_events) / NUM_TEST_ITERATIONS
        mlp_ms = sum(se.elapsed_time(ee) for se, ee in mlp_events) / NUM_TEST_ITERATIONS
        attn_core_ms = max(0.0, attn_ms - qkvo_ms)
    
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
        print(f"  [占比] QKVO: {qkvo_ms/total_ms*100:.1f}%, "
              f"Attention(core): {attn_core_ms/total_ms*100:.1f}%, "
              f"MLP: {mlp_ms/total_ms*100:.1f}%, "
              f"其他: {max(0.0, 100-(qkvo_ms+attn_core_ms+mlp_ms)/total_ms*100):.1f}%")
    
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
            'mlp_ms': mlp_ms
        })
    
    return result, tokens


def test_dense_pruned():
    """Dense (pruned) 配置测试 - 与 test_sparse 类似但不使用稀疏 kernel"""
    print("\n" + "=" * 80)
    print("测试: DENSE PRUNED (剪枝后权重 + dense GEMV)")
    print("=" * 80)
    
    qkvo_events, attn_events, mlp_events = [], [], []
    
    # 恢复原始 forward（权重已在优化阶段被替换为剪枝后权重）
    print("\n[准备] 恢复 dense forward 方法并加载剪枝后权重...")
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            proj_module = get_projection_module(layer, proj_name)
            proj_module.forward = original_forwards[layer_idx][proj_name]
            pruned_w = sparse_params[layer_idx][proj_name]['pruned_weight']
            proj_module.weight.data = pruned_w.T.to(DEVICE)
            # 注意：权重已在显存优化阶段被替换为剪枝后权重，无需重新加载
        layer.self_attn.forward = original_block_forwards[layer_idx]['self_attn']
        layer.mlp.forward = original_block_forwards[layer_idx]['mlp']
    print("✓ Dense forward 方法已恢复（使用已加载的剪枝后权重）")
    
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
    
    # 检查原始权重是否可用（显存优化后可能已被释放）
    print("\n[准备] 检查原始权重可用性...")
    first_layer_proj = list(sparse_params[0].keys())[0]
    if 'original_weight' not in sparse_params[0][first_layer_proj]:
        print("⚠️  原始权重已被释放（显存优化），跳过 dense_original 测试")
        print("   如需测试 dense_original，请重新运行脚本或关闭显存优化")
        return {
            'mode': 'dense_original', 
            'status': 'skipped_due_to_memory_optimization',
            'total_ms': 0, 'ms_per_token': 0, 'tokens_per_second': 0,
            'wall_total_ms': 0, 'wall_ms_per_token': 0, 'wall_tokens_per_second': 0
        }, []
    
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
    
    # 为 dense_original 添加模块级别的 NVTX 包装（已移除投影级别的细粒度标记）
    if PROFILE_COMPONENTS:
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            # 只保留模块级别的标记
            layer.self_attn.forward = _TimedForward(
                original_block_forwards[layer_idx]['self_attn'], None, "PY::self_attn"
            )
            layer.mlp.forward = _TimedForward(
                original_block_forwards[layer_idx]['mlp'], None, "PY::mlp"
            )
    
    # 为 dense_original 添加线性层的 NVTX 包装
    def make_dense_linear_wrapper(layer_idx, proj_name):
        """创建dense linear的NVTX包装器"""
        original_forward = original_forwards[layer_idx][proj_name]
        
        def wrapped_forward(x):
            nvtx.range_push(f"PY::dense_linear::layer{layer_idx}.{proj_name}")
            output = original_forward(x)
            nvtx.range_pop()  # PY::dense_linear::layerX.proj_name
            return output
        
        return wrapped_forward
    
    # 为所有线性层添加NVTX包装
    print("\n[准备] 为dense linear添加NVTX标记...")
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            proj_module = get_projection_module(layer, proj_name)
            proj_module.forward = make_dense_linear_wrapper(layer_idx, proj_name)
    print("✓ Dense linear NVTX标记已添加")
    
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
    
    # 使用 NVTX 捕获范围 + 可选的 torch profiler（可通过环境变量关闭）
    nvtx.range_push("CAPTURE")
    if os.getenv('E2E_DISABLE_TORCH_PROFILER', '0') != '1':
        torch.cuda.profiler.start()
    
    with torch.no_grad():
        for test_iter in range(NUM_TEST_ITERATIONS):
            decode_past = past_key_values
            decode_attn_mask = attention_mask.clone()
            decode_input_ids = cur_input_ids.clone()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            nvtx.range_push(f"DECODE_ITER_{test_iter}")
            start_event.record()
            
            for step in range(NUM_DECODE_STEPS):
                new_token_mask = torch.ones((batch_size, 1), device=decode_attn_mask.device, dtype=decode_attn_mask.dtype)
                
                cur_attn_mask = torch.cat([decode_attn_mask, new_token_mask], dim=1)
                
                nvtx.range_push("PY::model_forward")
                outputs = model(input_ids=decode_input_ids, past_key_values=decode_past,
                              attention_mask=cur_attn_mask, use_cache=True, return_dict=True)
                nvtx.range_pop()  # PY::model_forward
                
                decode_past = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                if test_iter == 0:
                    tokens.append(next_token.detach())
                next_token = next_token.unsqueeze(-1)
                
                decode_attn_mask = cur_attn_mask
                decode_input_ids = next_token
            
            end_event.record()
            nvtx.range_pop()  # DECODE_ITER_{test_iter}
            
            torch.cuda.synchronize()
            iter_time_ms = start_event.elapsed_time(end_event)
            iter_wall_ms = (time.perf_counter() - t0) * 1000.0
            all_times_ms.append(iter_time_ms)
            all_wall_times_ms.append(iter_wall_ms)
            if (test_iter + 1) % 10 == 0:
                print(f"  完成 {test_iter + 1}/{NUM_TEST_ITERATIONS} 次测试")
    
    # 停止 profiler / 关闭 CAPTURE 范围
    if os.getenv('E2E_DISABLE_TORCH_PROFILER', '0') != '1':
        torch.cuda.profiler.stop()
    nvtx.range_pop()
    
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
    output_dir = "/home/wangqitong/nmsparse_llama/logs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{TEST_MODE}_result.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 80)
