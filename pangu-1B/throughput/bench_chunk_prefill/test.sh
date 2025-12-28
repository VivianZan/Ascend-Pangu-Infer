HOST=127.0.0.1
FILE_NAME='result'

#no chunk
PORT1=8000

#chunk-2048
PORT2=8001

#chunk-8192
PORT3=8002

vllm bench serve   --model pangu_embedded_1b   --dataset-name random   --random-input-len 8192   --random-output-len 32   --num-prompts 100 --host $HOST --port $PORT1 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/  >> ${FILE_NAME}.txt 2>&1
vllm bench serve   --model pangu_embedded_1b   --dataset-name random   --random-input-len 8192   --random-output-len 32   --num-prompts 100 --host $HOST --port $PORT2 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/  >> ${FILE_NAME}.txt 2>&1
vllm bench serve   --model pangu_embedded_1b   --dataset-name random   --random-input-len 8192   --random-output-len 32   --num-prompts 100 --host $HOST --port $PORT3 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/  >> ${FILE_NAME}.txt 2>&1
