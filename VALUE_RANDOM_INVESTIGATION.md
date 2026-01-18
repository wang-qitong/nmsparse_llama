# write_balance_constant_v32 随机值问题深度调查

**研究者质疑**: 
1. `write_balance_constant_v32` 生成的 value 是随机值 `torch.rand(n, k_sparse)`，不是真实权重
2. index 只是按 bank/shuffle 生成，未检查 mask 或权重有效性
3. 文档声称是 Figure 9 精确格式，但数据是占位符

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🎯 核心发现

### **研究者的质疑完全正确！**

经过深入调查，我必须承认：

1. ✅ **`write_balance_constant_v32` 确实使用随机值**
2. ✅ **index 生成未使用真实权重位置**
3. ✅ **我的报告存在严重误导**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 第 1 部分：代码证据

### 1.1 write_balance_constant_v32（当前使用）

**路径**: `/root/SparTA/sparta/common/utils.py:857-911`

```python
def write_balance_constant_v32(dir_path, tesaid, name, state_dict, tesa, sparsity, align):
    import random 
    weight = state_dict['.'.join([name, 'weight'])].t()  # 读取了权重
    bias_key = ".".join([name, "bias"])
    if bias_key in state_dict:
        bias = state_dict[bias_key]
    else:
        bias = torch.zeros(weight.size(1), dtype=weight.dtype, device=weight.device)
    weight_tesa = tesa[tesaid]['weight'].t()  # 读取了 mask
    
    # ... 省略 ...
    
    k, n = weight.size()
    k_sparse = int(k * (1-sparsity))
    
    # ❌ 关键问题：使用随机值，完全忽略真实权重！
    value = torch.rand(n, k_sparse) # random value
    value = np.reshape(np.array(value), (n, k_sparse)).transpose().flatten().tolist()
    
    # ❌ 关键问题：index 生成与 mask/权重无关，仅基于 bank shuffle
    index = torch.zeros(n*k_sparse, dtype=torch.int32)
    t_index = torch.zeros(n*k_sparse, dtype=torch.int32)
    
    for j in range(0, n, align):
        for i in range(0, w, w//num_bank):
            random.shuffle(tmp_index)  # 随机 shuffle
            selected = tmp_index[:w//num_bank]
            selected = sorted(selected)
            
            # 生成 index，但未检查这些位置是否真的有非零值
            for k in range(0, w//num_bank):
                for j_in in range(0, align):
                    index[i+k + (j+j_in)*w] = selected[k]+ int(i/(1-sparsity))
                    t_index[(i+k)*n+j+j_in] = index[i+k+(j+j_in)*w]
    
    # 写入文件
    write_array(value, value_path, "f")      # 随机值！
    write_array(t_index, index_path)         # 随机 shuffle 的索引！
    write_array(bias, bias_path, "f")        # 只有 bias 是真实的
```

**关键问题**:
1. ❌ `weight` 和 `weight_tesa` 被读取但**从未使用**
2. ❌ `value = torch.rand(...)` 生成**随机数**
3. ❌ `index` 生成基于**随机 shuffle**，与真实稀疏 pattern 无关

### 1.2 write_balance_constant（旧版本，未使用）

**路径**: `/root/SparTA/sparta/common/utils.py:917-945`

```python
def write_balance_constant(dir_path, tesaid, name, state_dict, tesa, sparsity):
    weight = state_dict['.'.join([name, 'weight'])].t()
    bias = state_dict['.'.join([name, 'bias'])]
    weight_tesa = tesa[tesaid]['weight'].t()
    value = []
    index = []

    k, n = weight.size()
    k_sparse = int(k * (1-sparsity))
    
    # ✅ 正确：遍历真实权重和 mask
    for i in range(n):
        for j in range(k):
            if weight_tesa[j, i] \!= 0:  # 检查 mask
                value.append(weight[j, i])  # 使用真实权重值！
                index.append(j)             # 使用真实索引！

    value = np.reshape(np.array(value), (n, k_sparse)).transpose().flatten().tolist()
    index = np.reshape(np.array(index), (n, k_sparse)).transpose().flatten().tolist()
    
    write_array(value, value_path, "f")  # 真实权重
    write_array(index, index_path)       # 真实索引
```

**对比**:
| 方面 | write_balance_constant | write_balance_constant_v32 |
|------|----------------------|----------------------------|
| **Value** | ✅ 真实权重 | ❌ 随机值 |
| **Index** | ✅ 真实非零位置 | ❌ 随机 shuffle |
| **检查 mask** | ✅ 是 | ❌ 否 |
| **当前使用** | ❌ 已注释 | ✅ 在用 |

### 1.3 generate_balance_cfg 调用

**路径**: `/root/SparTA/sparta/common/utils.py:1070-1071`

```python
# write_balance_constant(constant_dir, tesaid, name, state, tesa, sparsity)  # 注释掉了
write_balance_constant_v32(constant_dir, tesaid, name, state, tesa, sparsity, align_n)  # 使用 v32
```

**结论**: 当前代码确实使用随机值版本！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 第 2 部分：为什么会这样？

### 2.1 可能的原因

#### 假设 1: 用于性能 benchmarking

```
场景: 测试 kernel 性能
  • 数据内容不重要（随机值即可）
  • 只需要正确的访问模式（conflict-free）
  • 快速生成测试数据
  
证据:
  ✅ Figure 12 是性能对比实验
  ✅ 不关心推理精度，只关心速度
  ✅ 随机值可以快速生成
```

#### 假设 2: 占位符，真实数据另外加载

```
场景: .bin 文件是占位符
  • 真实推理时从其他地方加载权重
  • NNFusion runtime 可能有其他机制
  
证据:
  ⚠️ 需要检查 NNFusion runtime 代码
  ⚠️ 需要检查 main_test.cpp 是否另外加载
```

#### 假设 3: 代码错误/未完成

```
场景: v32 是错误的实现
  • write_balance_constant 是正确版本
  • 但被注释掉改用 v32
  • 可能用于快速测试，忘记改回
  
证据:
  ✅ 代码中注释了正确版本
  ✅ v32 读取了 weight 但未使用（可疑）
```

### 2.2 BERT 流程验证

检查 BERT (Figure 8) 是否也这样：

**bert_large_balance_ck.py** (用于 BERT):
```python
# /root/nnfusion/Exp_Hardware/bert_large_balance_ck.py
# 最后几行:
export_tesa(norm_model.cpu(), data, outdir, mask)
generate_balance_cfg(outdir, align, total_m, sparsity_ratio)
```

调用同样的 `generate_balance_cfg`，所以**BERT 也使用随机值**！

### 2.3 关键线索

**llama2_large_balance_ck.py 的注释** (第 8 行):
```python
# 8) 按你的原逻辑：导出前把 Linear 权重全置为 1（仅用于 TESA/ONNX 导出）
for name, module in norm_model.named_modules():
    if isinstance(module, nn.Linear) and module.weight is not None:
        module.weight.data.fill_(1)  # 所有权重设为 1！
```

**重要发现**: 
- 导出前所有权重被设为 1
- 这说明**导出的权重本身就不是真实值**
- `write_balance_constant_v32` 使用随机值也就说得通了

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 第 3 部分：这意味着什么？

### 3.1 对 Figure 12 实验的影响

**Figure 12 目的**: 性能 benchmark

```
实验设置:
  • 比较不同稀疏度下的 kernel 性能
  • 不关心推理精度（没有准确率指标）
  • 只关心延迟（latency）
  
随机值的影响:
  ✅ 对性能测试无影响
  ✅ Kernel 执行时间不受数据值影响
  ✅ 只要访问模式正确即可
  
结论: Figure 12 使用随机值是合理的
```

### 3.2 对端到端推理的影响

**如果要做真实推理**:

```
❌ 当前流程不可行:
  • value.bin 是随机值
  • index.bin 是随机 shuffle 的索引
  • 无法产生正确的推理结果
  
✅ 必须修改:
  • 使用 write_balance_constant（旧版本）
  • 或重写 write_balance_constant_v32 使用真实权重
  • 确保 index 对应真实的非零位置
```

### 3.3 对 PyTorch Extension 方案的影响

**之前的方案假设错误**:

```
❌ 错误假设:
  • SparTA 生成的 .bin 文件包含真实权重
  • 可以直接加载用于推理
  
✅ 实际情况:
  • .bin 文件是随机值（用于性能测试）
  • 不能直接用于真实推理
  
✅ 正确方案:
  • 使用 write_balance_constant（旧版）
  • 或自己实现 condensed representation 转换
  • 确保使用真实权重和正确的索引
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 第 4 部分：修正后的理解

### 4.1 SparTA 的真实用途

```
SparTA/NNFusion 流程:
  
  Phase 1: 剪枝（llama2_large_balance_ck.py）
    ├─ 生成 N:M balanced mask
    ├─ 导出 ONNX（权重设为 1）
    └─ 生成 .bin 文件（随机值）
  
  Phase 2: NNFusion 编译
    ├─ 读取 ONNX 和 config
    ├─ 生成 CUDA kernel 代码
    └─ 编译为 executable
  
  Phase 3: 运行 main_test
    ├─ 初始化 (cuda_init)
    ├─ 可能加载真实权重？
    └─ 执行推理
  
目的: 性能 benchmarking，不是端到端推理
```

### 4.2 Figure 9 Kernel 的真实用途

```
Figure 9 (SpMV kernel):
  
  设计目的:
    • 展示 conflict-free access 优化
    • 论文中的性能对比实验
    • 不是完整的推理系统
  
  输入格式:
    ✅ mat_data [w, h] 列优先
    ✅ mat_index [w, h] 列优先
    ⚠️ 但论文实验中可能也用随机值测试
  
  真实推理:
    • 需要填充真实权重
    • 需要正确的索引
    • SparTA 当前流程未提供
```

### 4.3 Condensed Representation 的正确理解

```
论文 Figure 3 描述的格式:
  ✅ Data + Index 结构 - 正确
  ✅ 列优先存储 - 正确
  ✅ Conflict-free 优化 - 正确（index 生成逻辑）
  
  ❌ 我的错误理解:
    • 以为 SparTA 实现包含真实权重
    • 实际只是性能测试的占位符
  
  ✅ 正确理解:
    • Condensed representation 是格式定义
    • SparTA 的 v32 实现是性能测试版本
    • 真实推理需要使用 write_balance_constant（旧版）
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 第 5 部分：对 PyTorch Extension 方案的修正

### 5.1 不能直接使用 SparTA 的 .bin 文件

```
❌ 错误方案（之前的建议）:
  1. 运行 llama2_large_balance_ck.py
  2. 读取生成的 value_*.bin 和 index_*.bin
  3. 直接用于 PyTorch Extension
  4. 期望正确的推理结果
  
  问题: value 是随机值，无法正确推理
```

### 5.2 两种修正方案

#### 方案 A: 修改 SparTA 使用真实权重

```python
# 修改 sparta/common/utils.py 的 generate_balance_cfg

# 第 1071 行，改回使用旧版本:
# write_balance_constant_v32(...)  # 注释掉
write_balance_constant(constant_dir, tesaid, name, state, tesa, sparsity)  # 使用旧版

# 然后重新运行 llama2_large_balance_ck.py
```

**优点**:
- ✅ 使用真实权重
- ✅ index 对应真实非零位置
- ✅ 可用于真实推理

**缺点**:
- ⚠️ 没有 v32 的 conflict-free 优化
- ⚠️ index 生成逻辑不同

#### 方案 B: 自己实现完整的转换

```python
def generate_condensed_with_real_weights(
    weight,      # [n, k] 真实权重
    mask,        # [n, k] N:M balanced mask
    N, M, align_n
):
    """
    生成包含真实权重的 condensed representation
    结合 write_balance_constant 和 v32 的优点
    """
    n, k = weight.shape
    sparsity = 1 - N / M
    k_sparse = int(k * N / M)
    
    # 1. 生成 conflict-free index（来自 v32）
    bank_val = 32
    num_bank = k // bank_val
    tmp_index = list(range(k // num_bank))
    
    index = torch.zeros(n * k_sparse, dtype=torch.int32)
    
    for j in range(0, n, align_n):
        for i in range(0, k_sparse, k_sparse // num_bank):
            random.shuffle(tmp_index)
            selected = tmp_index[:k_sparse // num_bank]
            selected = sorted(selected)
            
            for kk in range(0, k_sparse // num_bank):
                for j_in in range(0, align_n):
                    idx = selected[kk] + int(i / (1 - sparsity))
                    index[(i + kk) * n + j + j_in] = idx
    
    # 2. 按 index 提取真实权重（修正！）
    value = torch.zeros(k_sparse * n, dtype=weight.dtype)
    weight_t = weight.t()  # [k, n]
    mask_t = mask.t()      # [k, n]
    
    for col in range(k_sparse):
        for row in range(n):
            idx_in_k = index[col * n + row].item()
            # ✅ 关键：检查 mask，使用真实权重
            if idx_in_k < k and mask_t[idx_in_k, row]:
                value[col * n + row] = weight_t[idx_in_k, row]
            else:
                # 如果 index 指向的位置不是非零，需要调整
                # 这是 v32 的问题：index 生成与 mask 不一致
                pass
    
    mat_data = value.reshape(k_sparse, n)
    mat_index = index.reshape(k_sparse, n)
    
    return mat_data, mat_index
```

**问题**:
- ⚠️ v32 的 index 生成与 mask 不一致
- ⚠️ 随机 shuffle 的 index 可能指向零值位置
- ⚠️ 需要重新设计 index 生成逻辑

#### 方案 C: 最佳方案 - 结合两者优点

```python
def generate_condensed_correct(
    weight,      # [n, k] 真实权重
    mask,        # [n, k] N:M balanced mask
    N, M, align_n
):
    """
    正确的 condensed representation 生成
    
    步骤:
    1. 从 mask 提取真实非零位置
    2. 使用 conflict-free 逻辑重新排列
    3. 填充真实权重值
    """
    n, k = weight.shape
    k_sparse = int(k * N / M)
    
    # 1. 提取每行的非零位置（来自 mask）
    weight_t = weight.t()  # [k, n]
    mask_t = mask.t()      # [k, n]
    
    nonzero_indices = []  # 每行的非零索引
    nonzero_values = []   # 每行的非零值
    
    for row in range(n):
        nz_idx = torch.nonzero(mask_t[:, row], as_tuple=False).squeeze(-1)
        nz_val = weight_t[nz_idx, row]
        nonzero_indices.append(nz_idx)
        nonzero_values.append(nz_val)
    
    # 2. 按 conflict-free 逻辑重新排列
    # TODO: 设计与 Figure 9 compatible 的排列
    # 保证 bank conflict-free 同时使用真实索引
    
    # 3. 生成 mat_data 和 mat_index
    ...
    
    return mat_data, mat_index
```

**挑战**:
- conflict-free 逻辑与真实稀疏 pattern 如何结合？
- 如果 mask 的非零位置不符合 conflict-free 要求怎么办？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 总结

### 研究者的质疑完全正确

1. ✅ **`write_balance_constant_v32` 使用随机值** - 证据确凿
2. ✅ **index 生成与真实权重无关** - 仅基于 random shuffle
3. ✅ **我的报告存在严重误导** - 假设 .bin 包含真实权重是错误的

### SparTA 的真实用途

- **目的**: 性能 benchmarking（Figure 12）
- **不是**: 端到端推理系统
- **value.bin**: 随机值占位符
- **index.bin**: conflict-free 访问模式，但与真实 mask 不一致

### 对 PyTorch Extension 方案的影响

**❌ 不能做的**:
- 直接使用 SparTA 生成的 .bin 文件
- 期望正确的推理结果

**✅ 必须做的**:
- 使用 `write_balance_constant`（旧版，真实权重）
- 或自己实现完整的 condensed representation 转换
- 确保 index 对应真实的非零位置
- 填充真实的权重值

### 关键教训

1. **不要盲目信任代码** - 即使是发表的论文
2. **性能测试 ≠ 真实推理** - 随机值足够测性能
3. **需要深入验证** - 特别是涉及真实推理时

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**感谢这位研究者的质疑！这暴露了我报告中的严重错误。**

