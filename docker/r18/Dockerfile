# Staged build context requirements are in docs/REPRODUCING.md.
ARG VLLM_BASE=vllm/vllm-openai:v0.34.0@sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484
FROM ${VLLM_BASE}
ARG VLLM_BASE
ARG MAX_JOBS=2
ARG BUILD_TRUE_RING=1
USER root
ENV PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 UV_OFFLINE=1 \
    HF_HUB_OFFLINE=1 TORCH_CUDA_ARCH_LIST=8.0 FLASHINFER_CUDA_ARCH_LIST=8.0 \
    MAX_JOBS=${MAX_JOBS} DSV41_TRUE_RING=0
WORKDIR /opt/dsv41
COPY tools/build_inputs.py tools/install_headers.py tools/check_exllama.py ./tools/
COPY offline/ ./offline/
RUN python3 tools/build_inputs.py "${VLLM_BASE}"
COPY vendor/exllamav3/ ./vendor/exllamav3/
ENV EXL3_EXT_INCLUDE=/opt/dsv41/vendor/exllamav3/exllamav3/exllamav3_ext
RUN python3 -m pip install --no-index --no-deps --no-build-isolation ./vendor/exllamav3 \
 && python3 tools/install_headers.py \
 && python3 tools/check_exllama.py \
 && rm -rf vendor/exllamav3/build
COPY vendor/vllm-exl3/ ./vendor/vllm-exl3/
COPY vendor/plugin-source.json ./vendor/plugin-source.json
COPY tools/verify_plugin.py tools/apply_patches.py tools/check_tc_build.py ./tools/
COPY patches/vllm-exl3/ ./patches/vllm-exl3/
COPY plugin-patched-files.json ./
RUN python3 tools/verify_plugin.py \
 && python3 tools/apply_patches.py vllm-exl3 \
 && python3 -m pip install --no-index --no-deps --no-build-isolation ./vendor/vllm-exl3 \
 && rm -rf vendor/vllm-exl3/build
RUN python3 vendor/vllm-exl3/src/vllm_exl3/tc_moe/build_standalone.py \
      --out /opt/dsv41-tc-moe --build-dir /tmp/dsv41-tc-build --verbose \
 && python3 tools/check_tc_build.py \
 && rm -rf /tmp/dsv41-tc-build
COPY native/coop/ /opt/dsv41-coop-kernel/
COPY tools/build_native.py tools/with_cuda_link_stub.sh ./tools/
RUN bash tools/with_cuda_link_stub.sh python3 tools/build_native.py coop
COPY base-files.json patched-files.json quantization_config.json series.json ./
COPY patches/vllm/ ./patches/vllm/
RUN python3 tools/apply_patches.py vllm
COPY patches/overlays/ ./patches/overlays/
COPY tools/apply_overlays.py ./tools/
RUN python3 tools/apply_overlays.py
COPY native/rowstore/ ./native/rowstore/
COPY native/ring/ /opt/dsv41/opera-true-ring/
COPY tools/build_row_store.py ./tools/
RUN python3 tools/build_native.py rowstore && python3 tools/build_row_store.py
RUN case "${BUILD_TRUE_RING}" in \
      0) ;; \
      1) FLASHINFER_WORKSPACE_BASE=/tmp/r18-fi bash tools/with_cuda_link_stub.sh \
           python3 tools/build_native.py ring && rm -rf /tmp/r18-fi ;; \
      *) echo 'BUILD_TRUE_RING must be 0 or 1' >&2; exit 2 ;; \
    esac
COPY adaptive-verify/ ./adaptive-verify/
COPY tools/ ./tools/
COPY tests/ ./tests/
COPY runtime/ ./runtime/
COPY vendor/OVERLAY-LICENSE vendor/VLLM-LICENSE vendor/EXLLAMAV3-PROVENANCE.md vendor/TC-MOE-PROVENANCE.json ./vendor/
COPY NOTICE.md ./
COPY overlays/ ./overlays/
COPY native/runtime/ ./native/runtime/
RUN python3 tools/install_overlays.py \
 && python3 tests/check_overlay_packages.py --image \
 && python3 tools/repin_overlays.py \
 && bash tools/with_cuda_link_stub.sh python3 tools/build_runtime_native.py
ENV VLLM_PLUGINS=vllm_exl3,dsv41_lora_argmax,dsv41_attn_port,dsv41_attn_audit,dsv41_rowstats,dsv41_wkv_replica,dsv41_dispatch_fuse,dsv41_ablit_r7,dsv41_indexer_bounded,dsv41_moe_stack,dsv41_prefill_7b,dsv41_fly_names
RUN python3 -m compileall -q tools tests \
 && python3 tools/check_api.py \
 && bash tools/with_cuda_link_stub.sh python3 tests/check_image_cpu.py --expect-ring "${BUILD_TRUE_RING}"
ENTRYPOINT ["vllm", "serve"]
