#include <cuda_runtime.h>
#include <torch/extension.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <vector>

#define NUM_BANK 128

// ===== 从 MV_one_kernel_block_batch.cu 复制的 initialData 函数 =====
void initialData(float *vec, float *mat_data, int *mat_index, 
                 float *mat_data_for_gpu, int *mat_index_for_gpu, 
                 int vecNum, int h, float sparse, int minibatch) {
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));
    unsigned int w = vecNum * sparse;

    // 生成随机输入向量
    for (int batch = 0; batch < minibatch; ++batch)
        for (int i = 0; i < vecNum; ++i) {
            vec[i + vecNum * batch] = (float)rand() / RAND_MAX;
        }

    // 生成随机稀疏矩阵值
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) {
            mat_data[i + j * w] = (float)rand() / RAND_MAX;
            mat_data_for_gpu[i * h + j] = mat_data[i + j * w];  // 列优先
        }

    // 生成 bank-aware 随机索引
    int* tmp_index = (int *)malloc(vecNum / NUM_BANK * sizeof(int));
    for (int i = 0; i < vecNum / NUM_BANK; ++i)
        tmp_index[i] = i;

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; i += w / NUM_BANK) {
            std::random_shuffle(tmp_index, tmp_index + vecNum / NUM_BANK);
            std::sort(tmp_index, tmp_index + w / NUM_BANK);
            for (int k = 0; k < w / NUM_BANK; ++k) {
                mat_index[i + k + j * w] = tmp_index[k] + i / sparse;
                mat_index_for_gpu[(i + k) * h + j] = mat_index[i + k + j * w];  // 列优先
            }
        }
    }
    free(tmp_index);
}

// ===== nmSPARSE EW GEMV Kernel (从 nmsparse_ew.cuh 复制) =====
extern "C" __global__ void nmsparse_ew_gemv_simt_fp32_fp32_fp32_32x32x32(
    float *g_vec, 
    float *g_mat_data, 
    int *g_mat_index, 
    float *g_odata, 
    int w, int h, 
    int BLOCK_WIDTH, 
    int NUM_THREADS, 
    int VEC_WIDTH, 
    const int minibatch, 
    const int vecNum
) {
    int blockxInd;
    int vecInd;
    int blockElt;
    
    if ((blockIdx.y + 1) * BLOCK_WIDTH <= w) {
        blockElt = BLOCK_WIDTH;
    } else {
        blockElt = w % BLOCK_WIDTH;
    }
    blockxInd = blockIdx.y * BLOCK_WIDTH;
    vecInd = blockIdx.y * VEC_WIDTH;
    
    unsigned int threadyInd = blockIdx.x * NUM_THREADS + threadIdx.x;
    extern __shared__ float vec_data[];
    
    // 加载输入向量到 shared memory
    #pragma unroll
    for (int batch = 0; batch < minibatch; ++batch) {
        #pragma unroll
        for (int i = 4 * threadIdx.x; i < VEC_WIDTH; i += 4 * NUM_THREADS) {
            *(float4 *)(vec_data + i + batch * VEC_WIDTH) = 
                *(float4 *)(g_vec + vecInd + i + (batch + blockIdx.z * minibatch) * vecNum);
        }
    }
    
    __syncthreads();
    
    float sdata[8] = {0};
    float data_tmp = 0;
    int index_tmp = 0;
    
    // 主计算循环
    #pragma unroll
    for (int index = 0; index < blockElt; ++index) {
        data_tmp = g_mat_data[threadyInd + (index + blockxInd) * h];
        index_tmp = g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd;
        
        #pragma unroll
        for (int batch = 0; batch < minibatch; batch += 1) {
            sdata[batch] += data_tmp * vec_data[index_tmp + batch * VEC_WIDTH];
        }
    }
    
    // 写回结果
    #pragma unroll
    for (int batch = 0; batch < minibatch; batch += 1) {
        atomicAdd(g_odata + h * (batch + blockIdx.z * minibatch) + threadyInd, sdata[batch]);
    }
}

// ===== PyTorch 包装函数 1: 调用 kernel =====
torch::Tensor nmsparse_spmv_forward(
    torch::Tensor vec,        // [M, K]
    torch::Tensor mat_data,   // [w, N]
    torch::Tensor mat_index,  // [w, N]
    int w, int h,
    int BLOCK_WIDTH, int NUM_THREADS, int VEC_WIDTH,
    int minibatch, int vecNum
) {
    // 创建输出 tensor
    auto output = torch::zeros({minibatch, h}, vec.options());
    
    // 计算 grid 和 block (与 Figure 9 等价)
    int BLOCK_minibatch = minibatch;
    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid(h / NUM_THREADS, w / BLOCK_WIDTH, minibatch / BLOCK_minibatch);
    
    int shared_mem_size = BLOCK_minibatch * VEC_WIDTH * sizeof(float);
    
    // 调用 kernel
    nmsparse_ew_gemv_simt_fp32_fp32_fp32_32x32x32<<<dimGrid, dimBlock, shared_mem_size>>>(
        vec.data_ptr<float>(),
        mat_data.data_ptr<float>(),
        mat_index.data_ptr<int>(),
        output.data_ptr<float>(),
        w, h, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch, vecNum
    );
    
    return output;
}

// ===== PyTorch 包装函数 2: 使用 initialData 生成数据 =====
std::vector<torch::Tensor> make_fig9_data(
    int vecNum,      // K, 输入维度
    int h,           // N, 输出维度
    float sparsity,  // 稀疏率
    int minibatch    // Batch size
) {
    float sparse = 1.0f - sparsity;
    int w = int(vecNum * sparse);
    
    // 1. 在 CPU 上分配内存
    float *h_vec = (float *)malloc(minibatch * vecNum * sizeof(float));
    float *h_mat_data = (float *)malloc(w * h * sizeof(float));
    int *h_mat_index = (int *)malloc(w * h * sizeof(int));
    float *h_mat_data_for_gpu = (float *)malloc(w * h * sizeof(float));
    int *h_mat_index_for_gpu = (int *)malloc(w * h * sizeof(int));
    
    // 2. 调用原版 initialData 生成数据
    initialData(h_vec, h_mat_data, h_mat_index, 
                h_mat_data_for_gpu, h_mat_index_for_gpu,
                vecNum, h, sparse, minibatch);
    
    // 3. 创建 PyTorch tensors 并拷贝数据
    auto options_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto options_i = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    
    // vec: [minibatch, vecNum]
    auto vec = torch::from_blob(h_vec, {minibatch, vecNum}, torch::kFloat32).to(torch::kCUDA);
    
    // mat_data: [w, h] (列优先，对应 mat_data_for_gpu)
    auto mat_data = torch::from_blob(h_mat_data_for_gpu, {w, h}, torch::kFloat32).to(torch::kCUDA);
    
    // mat_index: [w, h] (列优先，对应 mat_index_for_gpu)
    auto mat_index = torch::from_blob(h_mat_index_for_gpu, {w, h}, torch::kInt32).to(torch::kCUDA);
    
    // 4. 释放 CPU 内存
    free(h_vec);
    free(h_mat_data);
    free(h_mat_index);
    free(h_mat_data_for_gpu);
    free(h_mat_index_for_gpu);
    
    return {vec, mat_data, mat_index};
}

// ===== PYBIND11 模块定义 =====
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &nmsparse_spmv_forward, "nmSPARSE SpMV forward");
    m.def("make_fig9_data", &make_fig9_data, "Generate Fig9-style data using initialData");
}
