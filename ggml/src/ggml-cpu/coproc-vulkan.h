#pragma once

// CPU + GPU co-processing of Q4_0 matrix products (GGML_CPU_COPROC_VULKAN).
//
// On a system whose integrated GPU shares memory with the CPU, a prompt-sized Q4_0 matrix product is split by
// rows: the GPU computes the last rows of the weight matrix with a Vulkan kernel while the CPU threads compute
// the rest with the usual repacked kernels, both at the same time. The GPU's rows are copied into GPU buffers
// when the weights are loaded into the repack buffer.
//
// Environment:
//   GGML_CPU_COPROC=<fraction>     share of each Q4_0 weight's rows given to the GPU (0 or unset: off)
//   GGML_CPU_COPROC_MIN_N=<n>      use the GPU only for products with at least n columns (tokens), default 32
//   GGML_CPU_COPROC_DEVICE=<i>     Vulkan device index, default: the first device that is not a CPU
//   GGML_CPU_COPROC_STATS=1        print timing totals at exit (GPU time, time the CPU waited for the GPU)

#include "ggml.h"

struct ggml_cpu_coproc_weight;

// called when a Q4_0 tensor is loaded into the repack buffer, with its data in the standard Q4_0 layout
void ggml_cpu_coproc_prepare(const struct ggml_tensor * t, const void * data);

// the GPU part of a weight, or nullptr
struct ggml_cpu_coproc_weight * ggml_cpu_coproc_find(const struct ggml_tensor * t);

// number of rows (at the end of the weight matrix) the GPU computes for a product with n_cols columns, 0 = none
int64_t ggml_cpu_coproc_rows(const struct ggml_cpu_coproc_weight * w, int64_t n_cols);

// one product, called from forward_mul_mat:
//   thread 0: setup (buffers, whether the activations must be packed again); barrier;
//   all threads: pack (each a share of the columns); barrier; thread 0: submit (without waiting);
//   ... the CPU threads compute their rows ...
//   thread 0: wait; barrier; all threads: copy (each a share of the columns into dst)
void ggml_cpu_coproc_setup(struct ggml_cpu_coproc_weight * w, const struct ggml_tensor * src1);
void ggml_cpu_coproc_pack(struct ggml_cpu_coproc_weight * w, const struct ggml_tensor * src1, int ith, int nth);
void ggml_cpu_coproc_submit(struct ggml_cpu_coproc_weight * w);
void ggml_cpu_coproc_wait(struct ggml_cpu_coproc_weight * w);
void ggml_cpu_coproc_copy(struct ggml_cpu_coproc_weight * w, struct ggml_tensor * dst, int ith, int nth);
