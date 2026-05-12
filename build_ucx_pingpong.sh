#!/usr/bin/env bash
set -e
g++ -O2 -std=c++17 tools/ucx_pingpong.cpp -o bin/ucx_pingpong $(pkg-config --cflags --libs ucx)
echo "built bin/ucx_pingpong"
