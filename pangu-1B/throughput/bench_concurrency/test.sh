HOST=127.0.0.1
FILE_NAME='result'

#conc 1
PORT1=8000

#conc 8
PORT2=8001

#conc 16
PORT3=8002

vllm bench serve   --model pangu_embedded_1b   --dataset-name random --num-prompts 100 --host $HOST --port $PORT1 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/ >> ${FILE_NAME}.txt 2>&1
vllm bench serve   --model pangu_embedded_1b   --dataset-name random --num-prompts 100 --host $HOST --port $PORT2 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/ >> ${FILE_NAME}.txt 2>&1
vllm bench serve   --model pangu_embedded_1b   --dataset-name random --num-prompts 100 --host $HOST --port $PORT3 --trust-remote-code --tokenizer /opt/pangu/openPangu-Embedded-1B-V1.1/ >> ${FILE_NAME}.txt 2>&1