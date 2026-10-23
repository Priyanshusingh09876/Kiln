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
COPY csrc ./csrc
COPY setup.py ./
RUN pip install --no-build-isolation -e ".[train,ray,reward,triton]" \
 && python -c "from kiln import native; s = native.summary(); print(s); print('cuda ext built:', s['cuda_ext'])" \
 && (pip install flash-attn --no-build-isolation || true)
ENTRYPOINT ["kiln"]
