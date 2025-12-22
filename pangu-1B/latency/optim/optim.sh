FILE_NAME='optim'

export ASCEND_RT_VISIBLE_DEVICES=2

OUTPUT_LEN=128
BATCH_SIZE=1

for INPUT_LEN in 32 128 512 2048 8192
do
    echo "No optim" >> ${FILE_NAME}.txt
    echo "Batch-${BATCH_SIZE} Input len-${INPUT_LEN} Output len-${OUTPUT_LEN}" >> ${FILE_NAME}.txt
    vllm bench latency \
        --model /opt/pangu/openPangu-Embedded-1B-V1.1 \
        --num-iters 10 \
        --num-iters-warmup 5 \
        --trust-remote-code \
        --input-len $INPUT_LEN \
        --output-len $OUTPUT_LEN \
        --batch-size $BATCH_SIZE \
        --no-enable-chunked-prefill \
        --compilation-config '{"use_cudagraph":false}' \
        >> ${FILE_NAME}.txt 2>&1
done

for INPUT_LEN in 32 128 512 2048 8192
do
    echo "Optim" >> ${FILE_NAME}.txt
    echo "Batch-${BATCH_SIZE} Input len-${INPUT_LEN} Output len-${OUTPUT_LEN}" >> ${FILE_NAME}.txt
    vllm bench latency \
        --model /opt/pangu/openPangu-Embedded-1B-V1.1 \
        --num-iters 10 \
        --num-iters-warmup 5 \
        --trust-remote-code \
        --input-len $INPUT_LEN \
        --output-len $OUTPUT_LEN \
        --batch-size $BATCH_SIZE \
        --enable-chunked-prefill \
        --max-num-batched-tokens 4096 \
        --compilation-config '{"use_cudagraph":true}' \
        >> ${FILE_NAME}.txt 2>&1
done
