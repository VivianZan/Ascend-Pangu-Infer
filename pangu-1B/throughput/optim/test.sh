HOST=127.0.0.1

#no optim
PORT1=8000

#optim
PORT2=8001

for INPUT_LEN in 32 128 512 2048
do
    echo "No Optim: Input len-${INPUT_LEN}" >>  result.txt
    vllm bench serve   --model pangu_embedded_1b  --random-input-len $INPUT_LEN --num-prompts 100 --dataset-name sharegpt --dataset-path /opt/pangu/ShareGPT_V3_unfiltered_cleaned_split.json --host $HOST --port $PORT1 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/ >> result.txt 2>&1
done

for INPUT_LEN in 32 128 512 2048
do
    echo "Optim: Input len-${INPUT_LEN}" >>  result.txt
    vllm bench serve   --model pangu_embedded_1b  --random-input-len $INPUT_LEN --num-prompts 100 --dataset-name sharegpt --dataset-path /opt/pangu/ShareGPT_V3_unfiltered_cleaned_split.json --host $HOST --port $PORT2 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/ >> result.txt 2>&1
done
