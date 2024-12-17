ulimit -c unlimited  

export ZE_AFFINITY_MASK=2

# export TRITON_XPU_PROFILE=1 
export TORCHINDUCTOR_CACHE_DIR=${PWD}/torchinductor_cache/
export TRITON_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}/triton

# # export MLIR_ENABLE_DUMP=1
export TORCH_COMPILE_DEBUG=1
# # export TORCH_LOGS="+all"
# export TORCH_LOGS="+inductor"
# export TORCH_LOGS="+dynamo,+inductor"

# # export TORCHDYNAMO_REPRO_AFTER=dynamo
# # export TORCHDYNAMO_REPRO_AFTER=aot
# # export TORCHDYNAMO_REPRO_LEVEL=4

# # export TORCHINDUCTOR_MAX_AUTOTUNE=1

# # export TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL=s0,s1,s2

# export PYTORCH_ENABLE_XPU_FALLBACK=1
# # export PYTORCH_XPU_FALLBACK_OP=_adaptive_avg_pool2d_backward
# export PYTORCH_DEBUG_XPU_FALLBACK=1

# export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

# # export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1

# export IGC_ShaderDumpEnable=1 # dump asm code
# export IGC_DumpToCurrentDir=1 # dump asm code
# export NEO_CACHE_PERSISTENT=0

# export ONETRACE_EXECUTABLE=/home/yunfei/code/repo/pti-gpu/tools/onetrace/build/onetrace

# export FlushAllCaches=1

# export IPEX_ZE_TRACING=1

# export PYTORCH_DEBUG_XPU_FALLBACK=1

# # use 1 thread to for inductor compilation
# export TORCHINDUCTOR_COMPILE_THREADS=1

# # export TORCHINDUCTOR_BENCHMARK_KERNEL=1

# export UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0
# export IPEX_BN_ENG=BASIC  # ONEDNN / BASIC

# # export IGC_VISAOptions=" -TotalGRFNum 256 "

# rm -rf $TORCHINDUCTOR_CACHE_DIR

export TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1

# python -m pdb run_llama.py
python run_llama.py