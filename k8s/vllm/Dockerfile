# syntax=docker/dockerfile:1
# Standalone vLLM server for openai/gpt-oss-20b. Kept out of the main data-gatherer
# image on purpose: vLLM needs its own torch build, which would conflict with the
# main image's own stable torch stack used by everything else.
#
# Built on vLLM's own official image rather than a bare CUDA base + manual pip install --
# the official image bundles CUDA compatibility libraries for exactly the case where the
# node's driver is older than what the installed torch/vllm wheels target, which our own
# from-scratch build (nvidia/cuda base + `pip install vllm`) hit as a hard ImportError.
FROM --platform=linux/amd64 vllm/vllm-openai:latest

EXPOSE 8000
ENTRYPOINT ["vllm", "serve", "openai/gpt-oss-20b", \
    "--host", "0.0.0.0", "--port", "8000", \
    "--max-model-len", "131072", \
    "--gpu-memory-utilization", "0.85"]
