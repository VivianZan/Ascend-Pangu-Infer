DEVICE_INDEX="2"
export ASCEND_RT_VISIBLE_DEVICES=1

FILE_NAME='chunk_prefill'
INPUT_LEN=16384
OUTPUT_LEN=128
BATCH_SIZE=1

echo "Batch-${BATCH_SIZE} Input len-${INPUT_LEN} Output len-${OUTPUT_LEN}" >> ${FILE_NAME}.txt

echo "No Chunked Prefill" >> ${FILE_NAME}.txt
vllm bench latency \
    --model /opt/pangu/openPangu-Embedded-1B-V1.1 \
    --num-iters 10 \
    --num-iters-warmup 5 \
    --trust-remote-code \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --batch-size $BATCH_SIZE \
    --no-enable-chunked-prefill \
    >> ${FILE_NAME}.txt 2>&1

for INIT_CHUNK in 1024 2048 4096 8192
do
    echo "INIT_CHUNK: ${INIT_CHUNK}" >> ${FILE_NAME}.txt
    vllm bench latency \
        --model /opt/pangu/openPangu-Embedded-1B-V1.1 \
        --num-iters 10 \
        --num-iters-warmup 5 \
        --trust-remote-code \
        --input-len $INPUT_LEN \
        --output-len $OUTPUT_LEN \
        --batch-size $BATCH_SIZE \
        --enable-chunked-prefill \
        --max-num-batched-tokens $INIT_CHUNK \
        --long-prefill-token-threshold $INIT_CHUNK \
        >> ${FILE_NAME}.txt 2>&1
done

