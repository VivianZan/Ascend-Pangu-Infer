HOST=127.0.0.1

#no radix
PORT1=8001

#radix
PORT2=8000


# vllm bench serve   --model pangu_embedded_7b  --num-prompts 100 --dataset-name sharegpt --dataset-path /opt/pangu/ShareGPT_V3_unfiltered_cleaned_split.json --host $HOST --port $PORT1 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-7B-V1.1/ >> result.txt 2>&1
vllm bench serve   --model pangu_embedded_7b  --num-prompts 100 --dataset-name sharegpt --dataset-path /opt/pangu/ShareGPT_V3_unfiltered_cleaned_split.json --host $HOST --port $PORT2 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-7B-V1.1/ >> result.txt 2>&1
