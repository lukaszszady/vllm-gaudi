# `mixture_of_experts.mxfp4_fused_weights`: fixed per-local-expert cost and slow dequant in decode

## Summary

In decode with expert parallelism, `torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights` costs
**~0.56 ms per call even when no local expert receives a token**. On top of that it costs **~0.055 ms per
active local expert**, and almost all of that is `cast_packed_mxfp4_to_bf16`.

For Kimi-K3 (92 MoE layers) at decode batch 24, this amounts to:

- **~51 ms per step of fixed cost**;
- **~44 ms per step of dequant**;
- out of a **~205 ms step**.

About half of the fixed part is device idle time between serialized MME waves. Neither part can be removed
from the framework side without large memory or bandwidth costs.

We ask for:

1. skipping the MME and TPC work of local experts that receive no tokens;
2. an MXFP4 GEMM that dequantizes inside the kernel (or a substantially faster cast).

## Setup

| | |
|---|---|
| HW | 2 x 8 Gaudi3, one model replica, TP16 over Ray, expert parallel (EP16) |
| Framework | vLLM (v1) + vllm-gaudi, `torch.compile` with `hpu_backend` (no HPU graphs) |
| Op | `torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights(..., permuted_weights=True, activation="silu", experts_min=ep_rank*56, experts_max=ep_rank*56+55, is_fp4=True, block_size=32, chunk_size=0, total_experts=0)` |
| Weights | per local expert: w13 packed uint8 `[2*3072, 3584/2]` + E8M0 scales `[2*3072, 3584/32]`; w2 packed `[3584, 3072/2]` + scales `[3584, 3072/32]` (lists of 56 tensors) |
| Model | Kimi-K3: 896 routed experts, top_k 16, hidden 4096, routed-expert (latent) hidden 3584, moe_intermediate 3072, 92 MoE layers |
| Per rank | 56 local experts, ~17.5 MB of packed weights + scales per expert per layer |
| Decode | 24 tokens per step: 384 (token, expert) pairs over 896 experts, ~9-20 distinct local experts per rank per layer |

The call site is `vllm_gaudi/extension/ops.py` (`_mxfp4_fused_fwd`, `VllmMixtureOfExpertsOpMXFP4.forward`).

## Measurements

The data comes from a torch.profiler trace with HPU activities, one decode step, on all 16 ranks. Each
call compiles to its own recipe; its name contains `routed_experts` / `mixture_of_experts` / `MOE`. The
number of active local experts per call is taken as the count of `cast_packed_mxfp4_to_bf16` kernels
divided by 384.

### Cost model

The table is a linear fit over the 92 MoE calls of one step on one rank.

| Run | Active experts per call (mean, range) | Span per call | Busy (union of kernels) |
|---|---|---|---|
| A | 1.2 (0..5) | 0.557 ms + 0.0575 ms x active | 0.280 ms + 0.0608 ms x active |
| B | 9.0 (2..16) | 0.560 ms + 0.0548 ms x active | 0.301 ms + 0.0525 ms x active |

Kernel count per call is about 2020 + 552 x active.

### Call with 0 active local experts

This call has a span of 0.535 ms, is busy for 0.258 ms and runs 1992 kernels. There are 144 gaps
between kernels, of which 143 are shorter than 5 µs, adding up to 0.278 ms.

| Time from start (µs) | Kernels |
|---|---|
| 0 - 143 | routing / sort: `generate_bitonic_chunks_f32` x24, `router_stage1_i32` x14, `router_stage2_bf16` x14, **`router_stage3_bf16` x56**, `gather_elements`, `cumsum`, memgets, ... |
| 59 - 526 | **`GEMM` x912**: 114 waves of 8 GEMMs, one per MME, ~1.2 µs each. The waves are serialized, with ~2.8-3.6 µs between them. 114 = 2 x 56 + 2, so there is one wave for w13 and one for w2 of every local expert, even with no tokens routed to it |
| 526 - 535 | **`weighted_sum_reduction_bf16` x56** |

At ~3.4 µs per wave, the GEMM chain alone is ~0.39 ms per call, about 7 µs per local expert. The fixed
cost therefore scales with the number of local experts, not with the number of active ones.

Over 92 layers this is ~36 ms per step of empty MME waves. A further ~13 ms per step goes to
routing/sort, which is the same for every layer.

### Each active local expert

- 384 `cast_packed_mxfp4_to_bf16` kernels per expert, adding ~52-55 µs to the span.
- The expert's dequantized weights are 2 x 3072 x 3584 + 3584 x 3072 = 33 M values. That is 66 MB of bf16
  written plus ~17.5 MB of packed data read.
- This works out to ~1.6 TB/s of effective traffic. The GEMMs on the dequantized weights add only a few µs
  per expert.
- Every token of every step pays this materialization again, even though the weights never change.

### Effect on the step

At bs 24, a decode step is 205.6 ms, of which the routed MoE takes 96.8 ms.

- 92 x 0.56 ms ≈ 51 ms of fixed cost, of which ~25 ms is idle time inside the op.
- ~44 ms of cast.
- With EP, the slowest rank of every layer sets the pace, because the all_reduce follows the MoE. The
  max over ranks is ~1.55 x the mean number of active experts, which adds ~26 ms of all_reduce wait.

Fewer per-expert costs would shrink this wait as well.

## Why this cannot be fixed in the framework

- **Fewer local experts per call** would require gathering the active experts' weights into fixed-size
  slots, because shapes are static under torch.compile. That costs ~35 MB of HBM traffic per slot per
  layer, which is more than the ~7 µs of empty waves it would save.
- **More calls with smaller expert ranges** repeat the ~0.14 ms routing per call.
- **Keeping the weights in bf16 or FP8** takes 4x or 2x the ~84 GiB of expert weights per card, and does
  not fit.
- **Pure TP MoE (no EP)** would make every rank hold slices of all 896 experts, which multiplies the
  per-local-expert fixed cost by 16.

## Requests

1. **Skip inactive local experts.** Do not issue the MME waves, `router_stage3` and
   `weighted_sum_reduction` for experts whose token count is 0, for example with a device-side expert
   count and predicated or early-exit kernels. Alternatively, provide a compact mode that processes only
   the first N non-empty experts with a static N.
   - Expected gain: up to ~35 ms per step for this model.
2. **Lower the per-wave overhead.** Close the ~3 µs serialization between consecutive MME waves, e.g. by
   batching several experts per wave in decode, where tokens per expert are 1-8.
3. **MXFP4 GEMM with dequant in the kernel**, or a faster `cast_packed_mxfp4_to_bf16` that does not round
   trip through 66 MB of bf16 per expert.
   - Expected gain: most of the ~44 ms per step, scaling with the number of distinct experts. Real
     traffic hits about twice as many distinct experts as our dummy profile, so the gain is larger in
     serving.
4. **Document `chunk_size` / `total_experts`** and any env or flags that change the per-expert loop in
   decode: does `chunk_size` change the number of GEMM waves, and is there a dynamic variant that skips
   empty experts?

## Reproduction

Profiling is done with the vllm-gaudi dummy-batch profiler:

- `VLLM_PROFILE_DECODE=24,3648 VLLM_PROFILE_BLOCKS=scatter`
- plus `HPU_KIMI_SITU_IMPL=silu`, so that the fused MXFP4 op is used.

A standalone microbenchmark can be made from the setup table above. It needs:

- 56 experts with random MXFP4 weights;
- 24 tokens, top_k 16 over `total_experts=896`;
- `experts_min..experts_max` covering 0..20 of the routed ids;
- a sweep of the number of distinct routed local experts from 0 to 56.

Expected result: span ≈ 0.56 ms + 0.055 ms x active.
