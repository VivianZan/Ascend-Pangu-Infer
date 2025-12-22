FILE_NAME='io_len'

for BATCH_SIZE in 8
do 
    for INPUT_LEN in 32 64 128 256 512 1024
    do
        for OUTPUT_LEN in 32 64 128 256 512 1024
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
                --output-json  ${FILE_NAME}.json \
                >> ${FILE_NAME}.txt 2>&1
        done
    done
done
