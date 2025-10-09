#!/bin/bash
# Run trading evolution inside Docker

docker run --rm \
  -v ${PWD}/base_agent:/home/agent/agent_code:ro \
  -v ${PWD}/benchmark_data:/home/agent/benchmark_data:ro \
  -v ${PWD}/results/interactive_output:/home/agent/workdir:rw \
  sica_sandbox \
  python -m agent_code.agent trading-evolve -g 50 -f 100.0
