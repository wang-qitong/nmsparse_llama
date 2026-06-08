## 结论

不要在 hook 里直接做 CPU 外积。

原因是外积代价被严重低估：

```text
down_proj d_in = 27648
d_in × d_in = 764M elements
fp32 写入量 ≈ 3.0 GB / token / down_proj
64 层 down_proj ≈ 192 GB 写入 / token
再加 q/k/v/o/gate/up，CPU 内存带宽会成为绝对瓶颈
```

hook 是 forward 同步路径的一部分。Python hook 里的 `.cpu()` 和 `add_()` 会阻塞当前 decode step，不能被 GPU forward 完全掩盖。

---

## 当前问题

当前实现是：

```text
generate 一整条 prompt
  hook: 每层每个 decode token 把 activation 拷到 CPU，append 到 activation_buffers
generate 返回
  _flush_buffers_to_cpu_hessian()
  X.T @ X
  清空 activation_buffers
```

问题是 LongWriter 单条 prompt 可能生成几千到几万 token。

activation buffer 大小约：

```text
每 token ≈ 64 层 × 7 Linear activations ≈ 15 MB
1K token  ≈ 15 GB
5K token  ≈ 75 GB
10K token ≈ 150 GB
30K token ≈ 450 GB
```

再叠加：

```text
cpu_hessians 常驻 ≈ 203 GB / rank
KV cache 随 decode length 增长
GPU→CPU non_blocking copy 队列
```

因此长 prompt 会在 `generate()` 还没返回时 OOM。

---

## 推荐修改：分段 generate + 分段 flush

核心思路：

```text
不要一次 generate 到 max_new_tokens
改为每次 generate chunk_size 个 token
每个 chunk 结束后 flush activation_buffers
清空 activation_buffers 后继续下一段 generate
```

例如：

```text
chunk_size = 256 或 512

while total_new_tokens < max_new_tokens:
    generated = model.generate(
        input_ids=current_input_ids,
        max_new_tokens=chunk_size,
        ...
    )
    flush activation_buffers -> cpu_hessians
    current_input_ids = generated
    如果遇到 eos，break
```

这样 activation buffer 峰值从：

```text
O(整条 prompt 生成长度)
```

变成：

```text
O(chunk_size)
```

如果 `chunk_size=256`：

```text
256 × 15 MB ≈ 3.8 GB / rank
```

比 10K token 的 150 GB 安全很多。

---

## 为什么比 hook 内 CPU 外积更好

| 方案 | 优点 | 缺点 |
|---|---|---|
| hook 内 CPU 外积 | 不存 activation | 每 token 巨量 CPU 外积，decode 同步变慢 |
| prompt 结束 flush | GPU matmul 快 | 长 prompt activation 无限增长 |
| 分段 generate + 分段 flush | activation 有上限，仍用 GPU batched matmul | 每段会多一次 generate 调用和 flush |

推荐使用第三种。

---

## 实现要点

新增参数：

```text
--ar_flush_every_tokens 256
```

默认可以设为 `0` 保持旧行为，正式长生成时显式开启。

在 `ar_sparsegpt_sequential` 中替换单次：

```python
generated = model.generate(..., max_new_tokens=max_new)
```

为分段循环：

```python
remaining = max_new
generated = input_ids
while remaining > 0:
    step_new = min(flush_every, remaining)
    generated_next = model.generate(
        input_ids=generated,
        max_new_tokens=step_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones_like(generated),
    )
    new_tokens = generated_next.shape[1] - generated.shape[1]
    generated = generated_next

    if not mix_prefill_decode:
        _flush_buffers_to_cpu_hessian()

    remaining -= new_tokens
    if new_tokens == 0:
        break
    if tokenizer.eos_token_id is not None and generated[0, -1].item() == tokenizer.eos_token_id:
        break
```

注意：

```text
1. 每个 chunk 后必须 flush。
2. current input 继续用完整 generated，保证上下文连续。
3. 需要传 attention_mask，避免 pad_token == eos_token 的 warning。
4. 如果启用 first_k，可以在 collected token 达到 K 后停止 collecting，但生成可继续或直接停止，取决于需求。
5. last_k 模式不适合中途 flush，需要单独处理；建议先禁止 last_k + flush_every 组合。
```

---

## 推荐命令

两 rank，每 rank 两张卡：

```bash
python /wangqitong/sparsegpt-master_4/ar_sparsegpt.py \
    /wangqitong/qwen3-32b \
    longwriter_predjsonl \
    --ar_decoding \
    --prunen 2 --prunem 4 \
    --longwriter_jsonl /wangqitong/LongWriter/evaluation/models/qwen3-32b/pred.jsonl \
    --skip_eval \
    --ar_world_size 2 \
    --ar_gpus_per_rank 2 \
    --ar_flush_every_tokens 256 \
    --ar_hessian_dir /wangqitong/sparsegpt-master_4/outputs/hessians \
    --save_mask /wangqitong/sparsegpt-master_4/outputs/qwen3_32b_2_4.pt \
    --baseline_artifact_file /wangqitong/sparsegpt-master_4/outputs/artifacts.pt
```

如果用 5 rank 单卡并行，为避免单卡 KV cache 过长，仍建议加：

```text
--max_new_tokens 8192
```