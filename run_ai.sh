#!/bin/bash

echo "start for AI.." && source /workspace/dev.env/bin/activate
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
sleep 5
cd /workspace/AlphaPig/evaluate/ && echo "run AI is OK "
nohup python ChessClient.py --cur_role 1 --model r10 --room_name 1 --server_url http://gobang_server:8888 &
nohup python ChessClient.py --cur_role 1 --model r10 --room_name 2 --server_url http://gobang_server:8888 &
nohup python ChessClient.py --cur_role 1 --model r10 --room_name 3 --server_url http://gobang_server:8888 &
nohup python ChessClient.py --cur_role 1 --model r10 --room_name 4 --server_url http://gobang_server:8888 &
nohup python ChessClient.py --cur_role 1 --model r10 --room_name 5 --server_url http://gobang_server:8888 &
nohup python ChessClient.py --cur_role 2 --model r10 --room_name 6 --server_url http://gobang_server:8888 &
nohup python ChessClient.py --cur_role 2 --model r10 --room_name 7 --server_url http://gobang_server:8888 &
nohup python ChessClient.py --cur_role 2 --model r10 --room_name 8 --server_url http://gobang_server:8888 &
nohup python ChessClient.py --cur_role 2 --model r10 --room_name 9 --server_url http://gobang_server:8888 &
python ChessClient.py --cur_role 2 --model r10 --room_name 10 --server_url http://gobang_server:8888
