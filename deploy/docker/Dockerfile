# Trainer / rollout image. vLLM pins torch, so we build on the vLLM image and add Kiln.
ARG VLLM_VERSION=v0.8.5
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    NCCL_DEBUG=WARN NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=5 \
    TORCH_NCCL_AVOID_RECORD_STREAMS=1
WORKDIR /opt/kiln
COPY pyproject.toml README.md LICENSE ./
COPY kiln ./kiln
COPY configs ./configs
RUN pip install -e ".[train,ray,reward,triton]" && pip install flash-attn --no-build-isolation || true
ENTRYPOINT ["kiln"]
