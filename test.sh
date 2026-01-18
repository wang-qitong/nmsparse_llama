export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
hash -r
which nvcc
nvcc --version

# mkdir -p /wangqitong/.tmp /wangqitong/.torch_extensions
# export TMPDIR=/wangqitong/.tmp
# export TORCH_EXTENSIONS_DIR=/wangqitong/.torch_extensions
# export TORCH_CUDA_ARCH_LIST=8.0
# export MAX_JOBS=1

export CUDA_VISIBLE_DEVICES=0

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

# python /wangqitong/2_4/e2e_decode_modular.py
# python /wangqitong/4_8/e2e_decode_modular.py
# python /wangqitong/8_16/e2e_decode_modular.py
# python /wangqitong/16_32A800/e2e_decode_modular.py


# python /wangqitong/2_4/ppl.py --dataset wikitext2 --split test --modes dense_pruned --eval_style teacher_forcing --block_size 1024
# python /wangqitong/4_8/ppl_eval.py --dataset wikitext2 --split test --modes dense_pruned --eval_style teacher_forcing --block_size 1024
# python /wangqitong/8_16/ppl_eval.py --dataset wikitext2 --split test --modes dense_pruned --eval_style teacher_forcing --block_size 1024
# python /wangqitong/16_32A800/ppl_eval.py --dataset wikitext2 --split test --modes dense_pruned --eval_style teacher_forcing --block_size 1024
python /wangqitong/32_64/ppl_eval.py --dataset wikitext2 --split test --modes dense_pruned --eval_style teacher_forcing --block_size 1024