#!/usr/bin/env bash
# Quick launcher for vLLM UVM server
cd /tmp
PYTHON=/home/arush/data/vllm/nuvllm/.venv/bin/python
SCRIPT=/home/arush/data/ravi/uvm_access_tracking_1/start_vllm_server.py

exec $PYTHON $SCRIPT "$@"
