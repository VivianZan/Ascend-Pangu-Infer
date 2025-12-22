FILE_NAME='Npu_graph'

export ASCEND_RT_VISIBLE_DEVICES=2

INPUT_LEN=32
OUTPUT_LEN=128
BATCH_SIZE=1

echo "Batch-${BATCH_SIZE} Input len-${INPUT_LEN} Output len-${OUTPUT_LEN}" >> ${FILE_NAME}.txt

echo "No graph" >> ${FILE_NAME}.txt
vllm bench latency \
    --model /opt/pangu/openPangu-Embedded-7B-V1.1 \
    --num-iters 10 \
    --num-iters-warmup 5 \
    --trust-remote-code \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --batch-size $BATCH_SIZE \
    --no-enable-chunked-prefill \
    --compilation-config '{"use_cudagraph":false}' \
    >> ${FILE_NAME}.txt 2>&1

echo "Graph Mode" >> ${FILE_NAME}.txt
vllm bench latency \
    --model /opt/pangu/openPangu-Embedded-7B-V1.1 \
    --num-iters 10 \
    --num-iters-warmup 5 \
    --trust-remote-code \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --batch-size $BATCH_SIZE \
    --no-enable-chunked-prefill \
    --compilation-config '{"use_cudagraph":true}' \
    >> ${FILE_NAME}.txt 2>&1
