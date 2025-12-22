FILE_NAME='bench_quant'
INPUT_LEN=4096
OUTPUT_LEN=32
BATCH_SIZE=1

echo "Batch-${BATCH_SIZE} Input len-${INPUT_LEN} Output len-${OUTPUT_LEN}" >> ${FILE_NAME}.txt

vllm bench latency \
    --model /run/models/openPangu-1B-a8w8\
    --num-iters 5 \
    --num-iters-warmup 3 \
    --trust-remote-code \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --batch-size $BATCH_SIZE \
    --no-enable-chunked-prefill \
    --quantization ascend \
    >> ${FILE_NAME}.txt 2>&1

vllm bench latency \
    --model /opt/pangu/openPangu-Embedded-1B-V1.1\
    --num-iters 5 \
    --num-iters-warmup 3 \
    --trust-remote-code \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --batch-size $BATCH_SIZE \
    --no-enable-chunked-prefill \
    >> ${FILE_NAME}.txt 2>&1
