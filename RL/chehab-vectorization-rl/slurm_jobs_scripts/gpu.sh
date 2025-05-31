#!/bin/bash

srun --pty -n1 -q c2 -p nvidia  --mem=128G --gres=gpu:a100:2 -t 1-1 bash
