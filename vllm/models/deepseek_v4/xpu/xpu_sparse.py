# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU DeepSeek-V4 attention subclass.

Subclasses the shared ``DeepseekV4Attention`` ABC and provides XPU-native
Triton kernels for decode (FP8 dequant + BF16 attention) and prefill
(BF16 gathered KV + sparse attention).
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.models.deepseek_v4.xpu.xpu_sparse_decode_fp8 import (
    xpu_sparse_decode_fp8,
)
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4XPUSparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "XPU_V4_MLA_SPARSE"


class DeepseekV4XPUAttention(DeepseekV4Attention):
    """XPU sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4XPUSparseBackend
    use_flashmla_fp8_layout = True

    def __init__(self, *args, **kwargs) -> None:
        # torch.cuda.Event() raises RuntimeError on XPU ("dummy base class").
        # The Base and DeepseekV4Indexer both create cuda Events in __init__, so
        # we temporarily redirect torch.cuda.Event → torch.xpu.Event.
        _orig_event = torch.cuda.Event
        torch.cuda.Event = torch.xpu.Event  # type: ignore[misc]
        try:
            super().__init__(*args, **kwargs)
        finally:
            torch.cuda.Event = _orig_event  # type: ignore[misc]

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        from typing import cast

        if not isinstance(attn_metadata, dict):
            # Profile run: no-op, just return q (no padding needed on XPU).
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
            xpu_qnorm_rope_kv_fp8_insert,
        )

        xpu_qnorm_rope_kv_fp8_insert(
            q,
            kv,
            self.swa_cache_layer.kv_cache,
            swa_metadata.slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
        return q

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    @staticmethod
    def _maybe_upcast_e8m0(scale: torch.Tensor) -> torch.Tensor:
        if scale.dtype == torch.float8_e8m0fnu:
            return scale.to(torch.float32)
        return scale

    @classmethod
    def _reshape_wo_a_scale_to_bkn(
        cls,
        scale: torch.Tensor,
        n_local_groups: int,
        o_lora_rank: int,
        hidden_dim: int,
    ) -> torch.Tensor | None:
        # oneDNN batched path accepts weight scales as [B, N] (per-channel)
        # or [B, gK, gN] (block quant), where B is the group batch.
        if scale.ndim == 1 and scale.numel() == n_local_groups * o_lora_rank:
            return scale.view(n_local_groups, o_lora_rank).contiguous()

        if scale.ndim == 2:
            dim0, dim1 = scale.shape
            # Layout A: [G * gN, gK] -> [G, gK, gN]
            if dim0 % n_local_groups == 0:
                g_n = dim0 // n_local_groups
                g_k = dim1
                if (o_lora_rank % g_n == 0) and (hidden_dim % g_k == 0):
                    return scale.view(n_local_groups, g_n, g_k).transpose(1, 2).contiguous()
            # Layout B: [gK, G * gN] -> [G, gK, gN]
            if dim1 % n_local_groups == 0:
                g_n = dim1 // n_local_groups
                g_k = dim0
                if (o_lora_rank % g_n == 0) and (hidden_dim % g_k == 0):
                    return scale.view(g_k, n_local_groups, g_n).permute(1, 0, 2).contiguous()

        if scale.ndim == 3 and scale.shape[0] == n_local_groups:
            # [G, gK, gN]
            if (hidden_dim % scale.shape[1] == 0) and (o_lora_rank % scale.shape[2] == 0):
                return scale.contiguous()
            # [G, gN, gK] -> [G, gK, gN]
            if (hidden_dim % scale.shape[2] == 0) and (o_lora_rank % scale.shape[1] == 0):
                return scale.transpose(1, 2).contiguous()

        return None

    @classmethod
    def _get_cached_wo_a_fp8_bkn(
        cls,
        wo_a: torch.nn.Module,
        n_local_groups: int,
        o_lora_rank: int,
        hidden_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not hasattr(wo_a, "weight_scale_inv"):
            return None

        weight = wo_a.weight
        if weight.dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}:
            return None

        out_features = n_local_groups * o_lora_rank
        scale = wo_a.weight_scale_inv
        if scale.dtype == torch.float8_e8m0fnu:
            cached_scale = getattr(wo_a, "_dsv4_xpu_wo_a_scale_fp32", None)
            if cached_scale is None or cached_scale.shape != scale.shape:
                cached_scale = scale.to(torch.float32)
                wo_a._dsv4_xpu_wo_a_scale_fp32 = cached_scale
            scale = cached_scale
        else:
            scale = cls._maybe_upcast_e8m0(scale)
        scale_ptr = scale.data_ptr()
        cache_key = (weight.data_ptr(), scale_ptr, n_local_groups, o_lora_rank, hidden_dim)

        cached_key = getattr(wo_a, "_dsv4_xpu_wo_a_fp8_key", None)
        cached_val = getattr(wo_a, "_dsv4_xpu_wo_a_fp8_bkn", None)
        if cached_key == cache_key and cached_val is not None:
            return cached_val

        # wo_a can be either [G*D, R] (checkpoint/original) or [R, G*D]
        # (post-kernel canonicalized). Normalize both to [G, R, D].
        if weight.shape == (out_features, hidden_dim):
            weight_bkn = weight.view(n_local_groups, o_lora_rank, hidden_dim).transpose(1, 2)
        elif weight.shape == (hidden_dim, out_features):
            weight_bkn = weight.view(hidden_dim, n_local_groups, o_lora_rank).permute(1, 0, 2)
        else:
            return None

        scale_bkgn = cls._reshape_wo_a_scale_to_bkn(
            scale,
            n_local_groups=n_local_groups,
            o_lora_rank=o_lora_rank,
            hidden_dim=hidden_dim,
        )
        if scale_bkgn is None:
            return None

        cached = (weight_bkn.contiguous(), scale_bkgn)
        wo_a._dsv4_xpu_wo_a_fp8_key = cache_key
        wo_a._dsv4_xpu_wo_a_fp8_bkn = cached
        return cached

    def _o_proj_xpu_fp8(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor | None:
        if not (hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "fp8_gemm")):
            return None
        if not hasattr(self.wo_a, "weight_scale_inv"):
            return None

        # Generate FP8 activations with per-token-per-128 scales.
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=False,
        )

        hidden_dim = o_fp8.shape[-1]
        cached = self._get_cached_wo_a_fp8_bkn(
            self.wo_a,
            n_local_groups=self.n_local_groups,
            o_lora_rank=self.o_lora_rank,
            hidden_dim=hidden_dim,
        )
        if cached is None:
            return None

        wo_a_weight_bkn, wo_a_scale_bkgn = cached

        # fp8_gemm batched API: A [B, M, K], W [B, K, N] -> out [B, M, N].
        a_bmk = o_fp8.transpose(0, 1).contiguous()
        a_scale_bmgk = o_scale.transpose(0, 1).contiguous()
        z_gtd = torch.ops._xpu_C.fp8_gemm(
            a_bmk,
            wo_a_weight_bkn,
            torch.bfloat16,
            a_scale_bmgk,
            wo_a_scale_bkgn,
            None,
        )
        return z_gtd.transpose(0, 1).contiguous()

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        z = self._o_proj_xpu_fp8(o, positions)
        if z is not None:
            return self.wo_b(z.flatten(1))

        # Fallback to BF16 reference wo_a path (same as ROCm).
        from vllm.models.deepseek_v4.amd.rocm import rocm_inv_rope_einsum

        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        return self.wo_b(z.flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: reserve workspace, skip actual kernels.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        assert swa_indices is not None and swa_lens is not None
        xpu_sparse_decode_fp8(
            q=q,
            kv_cache=kv_cache,
            swa_kv_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            attn_sink=self.attn_sink,
            softmax_scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            out=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        chunk_size_const = self.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )

            kv_ws = kv[:chunk_size].reshape(-1, 1, q.shape[-1])
            out, _, _ = triton_bf16_mla_sparse_interface(
                q=q[query_start:query_end],
                kv=kv_ws,
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                d_v=q.shape[-1],
                block_dpe=0,
            )
            output[query_start:query_end] = out
