# 16_32A800/e2e_decode_modular.py 说明文档

本文档说明 `/wangqitong/16_32A800/e2e_decode_modular.py` 的整体流程、主要函数/类的功能，以及脚本开关（环境变量）如何使用。

## 1. 脚本用途（做什么）

- 对 `LLaMA-2-7B` 做端到端 decode 性能/正确性验证。
- 支持三种模式（通过环境变量 `E2E_MODE` 选择）：
  - `sparse`：使用 nmSPARSE kernel 走稀疏 GEMV
  - `dense_pruned`：使用剪枝后的权重，但走 dense GEMV
  - `dense_original`：使用原始权重，走 dense GEMV
- 支持一套“显存拆解”打印（通过 `E2E_MEM_REPORT` 打开），用于回答“显存都被什么占用”。

## 2. 快速运行

### 2.1 运行 sparse（默认模式）并打印显存拆解

```bash
E2E_MEM_REPORT=1 E2E_DISABLE_TORCH_PROFILER=1 python3 /wangqitong/16_32A800/e2e_decode_modular.py
```

### 2.2 切换模式

```bash
E2E_MODE=dense_pruned python3 /wangqitong/16_32A800/e2e_decode_modular.py
E2E_MODE=dense_original python3 /wangqitong/16_32A800/e2e_decode_modular.py
```

### 2.3 OOM sweep（逐步增加 decode steps 直到 OOM）

```bash
E2E_OOM_SWEEP=1 E2E_SWEEP_REPEATS=20 E2E_SWEEP_MAX_STEPS=4096 python3 /wangqitong/16_32A800/e2e_decode_modular.py
```

## 3. 关键参数（变量）与开关（环境变量）

### 3.1 脚本内常量（直接改代码生效）

- `LLAMA_MODEL_PATH`
  - 模型路径，默认 `/wangqitong/llama_2-7b`
- `DEVICE`
  - 默认 `cuda`
- `DTYPE`
  - 默认 `torch.float32`
- `PROMPT_TEXT`
  - 输入 prompt 文本
- `BATCH_SIZE`
  - 默认 `8`
- `SEQ_LEN`
  - 默认 `128`（prefill 的 token 数）
- `NUM_DECODE_STEPS`
  - 默认 `1`（每轮 decode 多少步）
- `NUM_TEST_ITERATIONS`
  - 默认 `100`（重复跑多少轮统计性能）
- `SPARSITY`
  - 稀疏率，默认 `0.5`
- `NUM_BANK_VAL`
  - bank size，默认 `32`
- `NUM_THREADS`
  - nmSPARSE kernel 线程数参数，默认 `128`
- `PROFILE_COMPONENTS`
  - 是否使用 CUDA Event 收集组件时间；默认 `False`

### 3.2 环境变量开关（运行时配置）

- `E2E_MODE`
  - **用途**：选择运行模式
  - **取值**：`sparse` / `dense_pruned` / `dense_original`
  - **默认**：`sparse`

- `E2E_MEM_REPORT`
  - **用途**：是否打印显存拆解（`[MEM] ...`）
  - **取值**：`1` 开 / `0` 关
  - **默认**：`1`

- `E2E_KEEP_PRUNED_WEIGHT_ON_GPU`
  - **用途**：是否把 `pruned_weight` 常驻搬到 GPU。
    - `sparse` 模式下 nmSPARSE forward 只需要 `mat_data/mat_index`，通常不需要 `pruned_weight` 常驻 GPU。
    - `dense_pruned` 模式会用到 `pruned_weight`。
  - **取值**：`1` 保留（更占显存） / `0` 不保留（省显存）
  - **默认**：`0`

- `E2E_DISABLE_TORCH_PROFILER`
  - **用途**：是否关闭 `torch.cuda.profiler.start/stop()`
  - **取值**：`1` 关闭 profiler / 其他值开启
  - **默认**：`0`（即默认开启；但为了避免额外开销通常建议设置为 `1`）

- `E2E_OOM_SWEEP`
  - **用途**：开启 OOM sweep 模式（逐步增加 decode steps 直到 OOM）
  - **取值**：`1` 开 / `0` 关
  - **默认**：`0`

- `E2E_SWEEP_REPEATS`
  - **用途**：OOM sweep 每个 step 重复测多少次
  - **默认**：`20`

- `E2E_SWEEP_MAX_STEPS`
  - **用途**：OOM sweep 最大 decode step 上限
  - **默认**：`4096`

## 4. 运行流程（从上到下）

1. **加载/编译 nmSPARSE CUDA 扩展**
   - 使用 `torch.utils.cpp_extension.load()` 编译 `nmsparse_wrapper_with_initialdata.cu` 并生成 `nmsparse_fig9.so`

2. **加载模型与 tokenizer**
   - `LlamaForCausalLM.from_pretrained(... device_map="cuda", torch_dtype=torch.float32)`

3. **加载或生成剪枝参数（sparse cache）**
   - 如果 `pruned_cache_file` 存在：`torch.load(..., map_location='cpu')`
   - 否则：逐层逐投影调用 `load_and_prune_weight()` + `convert_to_sparse_format()` 生成并缓存

4. **将稀疏参数搬到 GPU**
   - 必搬：`mat_data`、`mat_index`
   - 可选：`pruned_weight`（由 `E2E_KEEP_PRUNED_WEIGHT_ON_GPU` 控制）

5. **保存原始 forward 方法**
   - 保存每层每个投影的 `forward`，以及 `self_attn.forward` / `mlp.forward`，用于后续切换不同模式时恢复。

6. **Tokenize 输入**
   - 生成 `input_ids` / `attention_mask_tok`（搬到 GPU）

7. **按模式执行 test**
   - `sparse` / `dense_pruned` / `dense_original`

8. **保存输出 JSON**
   - 输出到 `/home/wangqitong/nmsparse_llama/logs/<mode>_result.json`

## 5. 函数与类说明（做什么功能）

### 5.1 显存统计相关（新增）

- `_format_gib(nbytes)`
  - 将字节数格式化成 GiB 字符串。

- `_tensor_nbytes(t)`
  - 返回 tensor 占用字节数：`numel * element_size`。

- `_sum_model_nbytes(model)`
  - 统计模型在 GPU 上的 `parameters()` + `buffers()` 占用总字节数。

- `_sum_sparse_params_nbytes(sparse_params)`
  - 统计 `sparse_params` 内在 GPU 上的张量占用。
  - 当前统计项：`mat_data`、`mat_index`、`pruned_weight`。

- `_sum_kv_cache_nbytes(past_key_values)`
  - 统计 `past_key_values`（KV cache）内各层 K/V tensor 在 GPU 上占用。

- `report_cuda_memory(tag, model=None, sparse_params=None, past_key_values=None)`
  - 打印 `[MEM] <tag>` 记录：
    - `torch.cuda.memory_allocated()` / `memory_reserved()` / `max_memory_allocated()`
    - `torch.cuda.mem_get_info()` 的 `free/total`
    - 分项估算：`model` / `sparse_params` / `kv_cache` / `other(estimated)`
  - 由 `E2E_MEM_REPORT` 控制开关。

### 5.2 稀疏 kernel 包装

- `class SparsifiedLinear`
  - **用途**：把某个 `nn.Linear` 的计算替换为 nmSPARSE kernel。
  - **输入**：`x` shape `[batch, seq, hidden]`。
  - **处理**：reshape 成二维 `[B*L, K]`，调用 `nmsparse_module.forward(x_2d, mat_data, mat_index, ...)`。
  - **输出**：再 reshape 回 `[batch, seq, out_features]`。

### 5.3 计时/包装类

- `class _TimedForward`
  - **用途**：用 CUDA Event 对一个 `forward` 做时间统计，并把 (start_event, end_event) 记录到列表。
  - 由 `PROFILE_COMPONENTS` 控制是否启用。

- `class _NVTXOnlyForward`
  - **用途**：保留一个最轻量的 wrapper 结构（当前实现只透传调用）。

### 5.4 投影模块选择

- `get_projection_module(layer, proj_name)`
  - **用途**：根据 `proj_name` 返回对应的线性层模块。
  - `q_proj/k_proj/v_proj/o_proj` 来自 `layer.self_attn`。
  - `gate_proj/up_proj/down_proj` 来自 `layer.mlp`。

### 5.5 Decode 测试工具函数

- `_run_decode_n_steps_and_time_last(...)`
  - **用途**：在已有 `past_key_values` 基础上 decode `decode_steps` 步，并只统计最后一步耗时。
  - 常用于 OOM sweep 每个 step 的单步延迟评估。

- `_oom_sweep_decode(...)`
  - **用途**：从 step=1 开始逐渐增加 decode steps，重复跑 `SWEEP_REPEATS` 次，直到遇到 OOM。
  - 返回每个 step 的统计（mean/p50/p90/p99 等）。

### 5.6 三种模式的测试入口

- `test_sparse()`
  - **用途**：稀疏模式端到端测试。
  - 关键步骤：
    - prefill（`model.model(... use_cache=True)`）生成 `past_key_values`
    - 将各投影 `forward` 替换为 `SparsifiedLinear`，走 nmSPARSE kernel
    - decode 循环跑 `NUM_TEST_ITERATIONS` 次，每次 `NUM_DECODE_STEPS` 步，统计耗时

- `test_dense_pruned()`
  - **用途**：dense GEMV + 剪枝权重（对照组）。
  - 关键步骤：恢复原始 forward；加载/使用剪枝权重；prefill + decode。

- `test_dense_original()`
  - **用途**：dense GEMV + 原始权重（对照组）。
  - 关键步骤：恢复原始权重；prefill + decode。
  - 脚本中会检查 `sparse_params` 里是否还保留 `original_weight`（有的情况下为了省显存可能被释放/不搬 GPU）。

### 5.7 主程序入口

- `if __name__ == '__main__':`
  - 根据 `E2E_MODE` 调用上述 `test_*()`
  - 将结果写入 JSON。

## 6. 常见问题（FAQ）

### 6.1 为什么显存很大？

`sparse` 模式显存大头通常来自：
- 模型权重（`model(params+buffers)`）
- 稀疏参数常驻（`mat_data` + `mat_index`，以及可选的 `pruned_weight`）
- KV cache（`past_key_values`）

你可以用 `E2E_MEM_REPORT=1` 来直接打印拆解。

### 6.2 如何在不改 dtype 的情况下减少显存？

优先级一般是：
1. 不让 `pruned_weight` 常驻 GPU（默认已这样做；用 `E2E_KEEP_PRUNED_WEIGHT_ON_GPU=0`）
2. 减少稀疏化范围（例如只做 QKVO，不做 MLP）
3. 压缩 `mat_index` 存储（需要改 kernel 与数据格式）
