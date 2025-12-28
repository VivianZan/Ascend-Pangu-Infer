export ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_USE_V1=1

HOST=127.0.0.1
PORT=8001

LOCAL_CKPT_DIR=/opt/pangu/openPangu-Embedded-1B-V1.1
SERVED_MODEL_NAME=pangu_embedded_1b

vllm serve $LOCAL_CKPT_DIR \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --max-num-seqs 16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 2048 \
    --dtype bfloat16 \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --enable-chunked-prefill