FILE_NAME='test_chunk'
export ASCEND_RT_VISIBLE_DEVICES=4
OUTPUT_LEN=1
BATCH_SIZE=1

for INPUT_LEN in 2048 4096 8192 16384
do
    echo "Batch-${BATCH_SIZE} Input len-${INPUT_LEN} Output len-${OUTPUT_LEN}" >> ${FILE_NAME}.txt
    vllm bench latency \
        --model /opt/pangu/openPangu-Embedded-7B-V1.1 \
        --num-iters 10 \
        --num-iters-warmup 5 \
        --trust-remote-code \
        --input-len $INPUT_LEN \
        --output-len $OUTPUT_LEN \
        --batch-size $BATCH_SIZE \
        --no-enable-chunked-prefill \
        >> ${FILE_NAME}.txt
done