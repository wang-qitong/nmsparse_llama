#!/bin/bash
# 三次独立运行 e2e_decode_modular.py，每次测试一种配置
# 进程退出后显存完全回收，避免碎片化
export CUDA_VISIBLE_DEVICES=3
GPU_ID=3
SCRIPT_PATH="/home/wangqitong/nmsparse_llama/2_4_sparsity/e2e_decode_modular.py"
LOG_DIR="/home/wangqitong/nmsparse_llama/2_4_sparsity/logs_mlp_2_4"
RESULTS_FILE="/home/wangqitong/nmsparse_llama/2_4_sparsity/e2e_results_summary_2_4.json"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "========================================"
echo "开始三次独立运行测试 (2:4 稀疏模式)"
echo "========================================"

# 1. Sparse 测试
echo ""
echo "【1/3】运行 Sparse 2:4 (pruned) 测试..."
E2E_MODE=sparse CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_PATH" 2>&1 | tee "${LOG_DIR}/sparse_2_4.log"
if [ $? -ne 0 ]; then
    echo "❌ Sparse 测试失败"
    exit 1
fi
echo "✓ Sparse 测试完成，进程已退出，显存已释放"
sleep 2

# 2. Dense (pruned) 测试
echo ""
echo "【2/3】运行 Dense (pruned) 测试..."
E2E_MODE=dense_pruned CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_PATH" 2>&1 | tee "${LOG_DIR}/dense_pruned_2_4.log"
if [ $? -ne 0 ]; then
    echo "❌ Dense (pruned) 测试失败"
    exit 1
fi
echo "✓ Dense (pruned) 测试完成，进程已退出，显存已释放"
sleep 2

# 3. Dense (original) 测试
echo ""
echo "【3/3】运行 Dense (original) 测试..."
E2E_MODE=dense_original CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_PATH" 2>&1 | tee "${LOG_DIR}/dense_original_2_4.log"
if [ $? -ne 0 ]; then
    echo "❌ Dense (original) 测试失败"
    exit 1
fi
echo "✓ Dense (original) 测试完成，进程已退出，显存已释放"

# 4. 汇总结果
echo ""
echo "========================================"
echo "汇总三次测试结果..."
echo "========================================"

python3 << 'EOF'
import json
import os

log_dir = "/home/wangqitong/nmsparse_llama/logs"
results = {}

# 读取三个结果文件
for mode in ['sparse', 'dense_pruned', 'dense_original']:
    result_file = f"{log_dir}/{mode}_result.json"
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results[mode] = json.load(f)

# 保存汇总结果
output_file = "/home/wangqitong/nmsparse_llama/e2e_results_summary.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

# 打印对比
print("\n【性能对比】")
if 'sparse' in results and 'dense_pruned' in results:
    sparse_ms = results['sparse']['total_ms']
    dense_p_ms = results['dense_pruned']['total_ms']
    speedup = dense_p_ms / sparse_ms
    print(f"  Sparse vs Dense (pruned) 加速比: {speedup:.3f}x")
    print(f"  Sparse: {sparse_ms:.3f} ms  |  Dense (pruned): {dense_p_ms:.3f} ms")

if 'sparse' in results and 'dense_original' in results:
    sparse_ms = results['sparse']['total_ms']
    dense_o_ms = results['dense_original']['total_ms']
    speedup = dense_o_ms / sparse_ms
    print(f"  Sparse vs Dense (original) 加速比: {speedup:.3f}x")
    print(f"  Sparse: {sparse_ms:.3f} ms  |  Dense (original): {dense_o_ms:.3f} ms")

print(f"\n✓ 汇总结果已保存到: {output_file}")
EOF

echo ""
echo "========================================"
echo "✓ 全部测试完成！"
echo "========================================"
echo "日志文件: ${LOG_DIR}/*.log"
echo "结果文件: ${RESULTS_FILE}"
