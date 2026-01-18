# 2:4稀疏模式实现（方案A）

## 概述

这个文件夹包含修改后的代码，实现2:4稀疏模式（而不是原来的16:32模式）。

## 修改内容

### 1. test_nmsparse.py (Line 50)
```python
# 原来: NUM_BANK_VAL = 32  (16:32稀疏)
# 修改: NUM_BANK_VAL = 4   (2:4稀疏)
NUM_BANK_VAL = 4
```

**影响**：
- NUM_BANK = K / 4 = 4096 / 4 = 1024 （从128增加到1024）
- BLOCK_WIDTH = w / NUM_BANK = 2048 / 1024 = 2 （从16减少到2）
- 每4个连续元素保留2个（50%稀疏率）

### 2. nmsparse_wrapper_with_initialdata.cu (Line 10)
```cuda
// 原来: #define NUM_BANK 128  (K/32)
// 修改: #define NUM_BANK 1024 (K/4)
#define NUM_BANK 1024
```

**说明**：必须与test_nmsparse.py中的NUM_BANK保持一致。

### 3. nmsparse_test_utils.py
**无需修改** - 该文件使用参数化的num_bank_val，会自动适配2:4模式。

## 使用方法

### 测试2:4稀疏模式

```bash
cd /home/wangqitong/nmsparse_llama/2_4_sparsity

# 使用conda环境
conda activate myenv

# 运行测试
python test_nmsparse.py
```

### 与16:32模式对比

```bash
# 16:32模式（原版）
cd /home/wangqitong/nmsparse_llama
python test_nmsparse.py

# 2:4模式（新版）
cd /home/wangqitong/nmsparse_llama/2_4_sparsity
python test_nmsparse.py
```

## 预期效果

### 参数变化
| 参数 | 16:32模式 | 2:4模式 |
|------|----------|---------|
| NUM_BANK_VAL | 32 | 4 |
| NUM_BANK | 128 | 1024 |
| BLOCK_WIDTH | 16 | 2 |
| VEC_WIDTH | 32 | 4 |
| Grid blocks (y) | 128 | 1024 |

### 稀疏模式对比
- **16:32**: 每32个连续元素保留16个（50%）
- **2:4**: 每4个连续元素保留2个（50%）

### 性能预期
- **精度**: 可能略有提升（更细粒度控制）
- **速度**: 可能略有下降（grid配置次优，BLOCK_WIDTH=2太小）
- **Bank Conflict**: 仍然保证conflict-free（索引仍排序）

## 注意事项

1. **K维度必须是4的倍数**
   - 当前：K=4096 ✓
   - 如果K不是4的倍数，需要padding或调整

2. **NUM_BANK定义一致性**
   - test_nmsparse.py中的NUM_BANK计算
   - 必须与nmsparse_wrapper_with_initialdata.cu中的#define NUM_BANK一致

3. **性能评估**
   - 建议对比16:32和2:4的性能
   - 评估精度提升是否值得性能下降

## 文件清单

```
2_4_sparsity/
├── README.md                            # 本文件
├── test_nmsparse.py                     # 主测试脚本（已修改NUM_BANK_VAL=4）
├── nmsparse_wrapper_with_initialdata.cu # CUDA wrapper（已修改NUM_BANK=1024）
└── nmsparse_test_utils.py               # 工具函数（无需修改）
```

## 验证步骤

运行测试后检查：
1. 输出中`NUM_BANK=1024` 和 `BLOCK_WIDTH=2`
2. 索引分布验证通过（Bank conflict-free）
3. 正确性验证通过（误差<1e-4）
4. 对比sparse vs dense的加速比

## 回退到16:32

如果需要回退，只需使用原版文件：
```bash
cd /home/wangqitong/nmsparse_llama
python test_nmsparse.py
```

或者修改2_4_sparsity中的文件：
- test_nmsparse.py: `NUM_BANK_VAL = 32`
- nmsparse_wrapper_with_initialdata.cu: `#define NUM_BANK 128`
