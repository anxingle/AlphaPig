#!/bin/bash

echo "start...2" && source /workspace/dev.env/bin/activate && \
cd /workspace/AlphaPig/evaluate/ && echo "run server is OK " \
&& python ChessServer.py  --port 8888