HOST=127.0.0.1

#no radix
PORT1=8001

#radix
PORT2=8000


# vllm bench serve   --model pangu_embedded_1b  --num-prompts 100 --dataset-name sharegpt --dataset-path /opt/pangu/ShareGPT_V3_unfiltered_cleaned_split.json --host $HOST --port $PORT1 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/ >> result.txt 2>&1
# vllm bench serve   --model pangu_embedded_1b  --num-prompts 100 --dataset-name sharegpt --dataset-path /opt/pangu/ShareGPT_V3_unfiltered_cleaned_split.json --host $HOST --port $PORT2 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/ >> result.txt 2>&1


vllm bench serve   --model pangu_embedded_1b  --num-prompts 100 --random-prefix-len 2048 --random-input-len 1024  --random-output-len 32 --dataset-name sharegpt --dataset-path /opt/pangu/ShareGPT_V3_unfiltered_cleaned_split.json --host $HOST --port $PORT1 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/ >> result.txt 2>&1
vllm bench serve   --model pangu_embedded_1b  --num-prompts 100 --random-prefix-len 2048 --random-input-len 1024  --random-output-len 32 --dataset-name sharegpt --dataset-path /opt/pangu/ShareGPT_V3_unfiltered_cleaned_split.json --host $HOST --port $PORT2 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/ >> result.txt 2>&1
