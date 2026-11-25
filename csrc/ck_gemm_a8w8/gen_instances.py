# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_CORE_DIR = (
    os.path.join(os.path.abspath(f"{this_dir}/../../../"), "aiter/jit/utils")
    if os.path.exists(
        os.path.join(os.path.abspath(f"{this_dir}/../../../"), "aiter_meta")
    )
    else os.path.abspath(f"{this_dir}/../../aiter/jit/utils")
)
sys.path.insert(0, AITER_CORE_DIR)
from chip_info import (  # noqa: E402
    build_tune_dict,
    get_build_targets,
    write_lookup_header,
)
from gemm_a8w8_common import (  # noqa: E402
    default_kernels_dict,
    kernelInstance,
    kernels_list,
)


def _instance_filters(istune):
    """Resolve the gfx908 instance-pruning filters.

    Returns (kernels_filter, dtype_filter):

    - kernels_filter: None (build all kernels) or a set of kernelIds to keep.
      Sourced from the AITER_CK_INSTANCE_LIST file (one kernelId per line,
      '#'-comments allowed; absent/empty-file = all, preserving upstream
      behavior). Also accepts comma-separated ids via AITER_CK_INSTANCE_LIST
      itself for short lists.
    - dtype_filter: None (emit all 8 dtype combos, upstream behavior) or a
      set of (ABDtype, DDtype, EDtype) tuples to emit. Activated when every
      build target is gfx908 (MI100: no fp8 datapath, fp16-serving stack),
      or overridden via AITER_CK_DTYPES as comma-separated
      ABxDxD triplets, e.g. "I8xF32xF16,I8xF16xF16".
    """
    kernels_filter = None
    list_env = os.getenv("AITER_CK_INSTANCE_LIST", "")
    if list_env:
        if os.path.isfile(list_env):
            ids = set()
            with open(list_env) as f:
                for line in f:
                    tok = line.split("#", 1)[0].strip()
                    if tok:
                        ids.update(int(x) for x in tok.split(",") if x.strip())
        else:
            ids = {
                int(x) for x in list_env.split(",") if x.strip()
            }
        unknown = ids - set(kernels_list)
        if unknown:
            raise SystemExit(
                f"AITER_CK_INSTANCE_LIST references unknown kernelIds {sorted(unknown)}; "
                f"kernels_list has ids 0..{len(kernels_list) - 1}"
            )
        if ids:
            kernels_filter = ids

    dtype_filter = None
    dtypes_env = os.getenv("AITER_CK_DTYPES", "")
    if dtypes_env:
        dtype_filter = set()
        for tok in dtypes_env.split(","):
            ab, d, e = (x.strip() for x in tok.split("x"))
            dtype_filter.add((ab, d, e))
    elif all(gfx == "gfx908" for gfx, _ in get_build_targets()):
        # gfx908 default: int8 activations/weights (no fp8 datapath on MI100).
        # Both epilogues are kept: the target model serves fp16 (--dtype half)
        # but the DFlash2 draft resolves bf16 from its checkpoint and runs
        # <I8, F32, B16> GEMMs. Both compute variants (fp32/fp16 scale math)
        # stay -- fp32 is the accuracy hedge (see GFX908_BUILD_PLAN.md Step 5).
        dtype_filter = {
            ("I8", "F32", "F16"),
            ("I8", "F16", "F16"),
            ("I8", "F32", "B16"),
            ("I8", "B16", "B16"),
        }

    return kernels_filter, dtype_filter


class gemm_a8w8_fwd_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune

    def gen_instance(self, k: kernelInstance):
        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"

template <typename ABDataType, typename DDataType, typename EDataType = DDataType>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch)
{{{{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
    bool pad = (M % {k.MPerBLOCK} != 0) || (N % {k.NPerBLOCK} != 0) || (K % ({k.KPerBLOCK} * KBatch) != 0);
    using AccDataType = std::conditional_t<ck::is_same_v<ABDataType, I8>, I32, F32>;
    if (pad)
    {{{{
        // pad
        {{INSTANCE_CONTENT_pad}}
        // pad
    }}}}
    else
    {{{{
        // no pad
        {{INSTANCE_CONTENT_nopad}}
        // no pad
    }}}}
}}}}

"""
        INSTANCE_CONTENT_bias = f"""if (bias != std::nullopt)
        {{{{
            using DeviceGemmInstance = DeviceGemmHelper<
                ABDataType,
                AccDataType,
                DDataType, EDataType,
                MultiplyMultiplyAdd<AccDataType, DDataType, EDataType>,
                {k.BLOCK_SIZE},
                {k.MPerBLOCK},
                {k.NPerBLOCK},
                {k.KPerBLOCK},
                {k.WAVE_TILE_M},
                {k.WAVE_TILE_N},
                {k.WAVE_MAP_M},
                {k.WAVE_MAP_N},
                S<{(", ").join(str(x) for x in k.ABLOCK_TRANSFER)}>,
                S<{(", ").join(str(x) for x in k.BBLOCK_TRANSFER)}>,
                S<{(", ").join(str(x) for x in k.CBLOCK_TRANSFER)}>,
                S<{(", ").join(str(x) for x in k.CBLOCK_SPV)}, {k.CBLOCK_SPV[0]}>,
                {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
                {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
                ck::BlockGemmPipelineScheduler::{k.LOOP_SCHED},
                ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
                ck::tensor_operation::device::GemmSpecialization::{{GemmSpec}}>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<ABDataType, DDataType, EDataType, true, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
        }}}}
        else
        {{{{
            using DeviceGemmInstance = DeviceGemmHelper<
                ABDataType,
                AccDataType,
                DDataType, EDataType,
                RowwiseScale<AccDataType, DDataType, EDataType>,
                {k.BLOCK_SIZE},
                {k.MPerBLOCK},
                {k.NPerBLOCK},
                {k.KPerBLOCK},
                {k.WAVE_TILE_M},
                {k.WAVE_TILE_N},
                {k.WAVE_MAP_M},
                {k.WAVE_MAP_N},
                S<{(", ").join(str(x) for x in k.ABLOCK_TRANSFER)}>,
                S<{(", ").join(str(x) for x in k.BBLOCK_TRANSFER)}>,
                S<{(", ").join(str(x) for x in k.CBLOCK_TRANSFER)}>,
                S<{(", ").join(str(x) for x in k.CBLOCK_SPV)}>,
                {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
                {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
                ck::BlockGemmPipelineScheduler::{k.LOOP_SCHED},
                ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
                ck::tensor_operation::device::GemmSpecialization::{{GemmSpec}}>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<ABDataType, DDataType, EDataType, false, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
        }}}}
"""
        INSTANCE_CONTENT_nobias = f"""using DeviceGemmInstance = DeviceGemmHelper<
            ABDataType,
            AccDataType,
            DDataType, EDataType,
            RowwiseScale<AccDataType, DDataType, EDataType>,
            {k.BLOCK_SIZE},
            {k.MPerBLOCK},
            {k.NPerBLOCK},
            {k.KPerBLOCK},
            {k.WAVE_TILE_M},
            {k.WAVE_TILE_N},
            {k.WAVE_MAP_M},
            {k.WAVE_MAP_N},
            S<{(", ").join(str(x) for x in k.ABLOCK_TRANSFER)}>,
            S<{(", ").join(str(x) for x in k.BBLOCK_TRANSFER)}>,
            S<{(", ").join(str(x) for x in k.CBLOCK_TRANSFER)}>,
            S<{(", ").join(str(x) for x in k.CBLOCK_SPV)}>,
            {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
            {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
            ck::BlockGemmPipelineScheduler::{k.LOOP_SCHED},
            ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
            ck::tensor_operation::device::GemmSpecialization::{{GemmSpec}}>;
        // Run kernel instance.
        return gemm_a8w8_rowwise_impl<ABDataType, DDataType, EDataType, false, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
"""
        if self.istune:
            INSTANCE_IMPL_str = INSTANCE_IMPL.format(
                INSTANCE_CONTENT_pad=(
                    INSTANCE_CONTENT_nobias.format(GemmSpec="MNKPadding")
                ),
                INSTANCE_CONTENT_nopad=(
                    INSTANCE_CONTENT_nobias.format(GemmSpec="Default")
                ),
            )
        else:
            INSTANCE_IMPL_str = INSTANCE_IMPL.format(
                INSTANCE_CONTENT_pad=INSTANCE_CONTENT_bias.format(
                    GemmSpec="MNKPadding"
                ),
                INSTANCE_CONTENT_nopad=INSTANCE_CONTENT_bias.format(GemmSpec="Default"),
            )

        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(
            INSTANCE_IMPL_str
        )

        INSTANCE_template = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/{name}.cuh"

template torch::Tensor
{name}<{dtypes}>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch);

"""
        if self.istune:
            # Generate tune instances. I8 gets both epilogues: eB16 (upstream
            # default) and eF16 with dF32/dF16 scale math. eB16 is reachable
            # with both scale maths: dB16 (upstream default) and dF32 (fp32
            # scales + bf16 out, dispatched by gemm_a8w8.cu for bf16
            # outputs), so tuning on eB16-only would measure the wrong
            # scale math. F8 stays eB16-only.
            for EDtype, DDtype in [
                ("B16", "B16"),
                ("F16", "F32"),
                ("F16", "F16"),
                ("B16", "F32"),
            ]:
                INSTANCE_abI8 = INSTANCE_template.format(
                    name=k.name, dtypes=f"I8, {DDtype}, {EDtype}"
                )
                Path(
                    os.path.join(
                        self.instances_path,
                        f"{k.name}_abI8_d{DDtype}_e{EDtype}.cpp",
                    )
                ).write_text(INSTANCE_abI8)

            # F8 instances -- skipped under dtype pruning (gfx908 has no fp8
            # datapath; an int8-only sweep never calls them)
            _df = getattr(self, "dtype_filter", None)
            if _df is None or any(ab == "F8" for ab, _, _ in _df):
                for EDtype in ["B16"]:
                    INSTANCE_abF8 = INSTANCE_template.format(
                        name=k.name, dtypes=f"F8, F32, {EDtype}"
                    )
                    Path(
                        os.path.join(
                            self.instances_path, f"{k.name}_abF8_dF32_e{EDtype}.cpp"
                        )
                    ).write_text(INSTANCE_abF8)
        else:
            # combos as (ABDtype, DDtype, EDtype) to match _instance_filters
            combos = [
                (ABDtype, DDtype, EDtype)
                for EDtype in ["B16", "F16"]
                for ABDtype in ["I8", "F8"]
                for DDtype in ["F32", EDtype]
            ]
            dtype_filter = getattr(self, "dtype_filter", None)
            if dtype_filter is not None:
                combos = [c for c in combos if c in dtype_filter]
            for ABDtype, DDtype, EDtype in combos:
                intsance = INSTANCE_template.format(
                    name=k.name, dtypes=f"{ABDtype}, {DDtype}, {EDtype}"
                )
                Path(
                    os.path.join(
                        self.instances_path,
                        f"{k.name}_ab{ABDtype}_d{DDtype}_e{EDtype}.cpp",
                    )
                ).write_text(intsance)

    def gen_lookup_dict(self, kernels_dict):
        LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(ABTYPE, DTYPE, ETYPE)                                                                                      \\
   {                                                                                                                             \\"""

        LOOKUP_template = """
       {{{MNK},                                                                                                       \\
        {kernel_name}<ABTYPE, DTYPE, ETYPE>}},                       \\"""

        LOOKUP_end = """
   }

#endif // USE_ROCM
"""
        write_lookup_header(
            os.path.join(self.working_path, "gemm_a8w8_lookup.h"),
            kernels_dict,
            LOOKUP_head,
            LOOKUP_template,
            LOOKUP_end,
            self.istune,
        )

    def gen_manifest_head(self, kernels_dict):
        MAINFEST_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>
"""
        MAINFEST_template = """
template <typename ABDataType, typename DDataType, typename EDataType>
torch::Tensor
{kernel_name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch);
"""
        MAINFEST_end = """

#endif // USE_ROCM
"""

        with open(os.path.join(self.working_path, "gemm_a8w8_manifest.h"), "w") as f:
            f.write(MAINFEST_head)
            f.writelines(
                MAINFEST_template.format(kernel_name=k.name)
                for mnk, k in kernels_dict.items()
            )
            f.write(MAINFEST_end)

    def gen_built_combos_header(self, dtype_filter):
        """Write gemm_a8w8_built_combos.h declaring which (AB,D,E) dtype
        combos have instances in this build. gemm_a8w8.cu compiles out the
        dispatch branches for pruned combos (TORCH_CHECK fallback) -- an
        unguarded branch would reference never-emitted kernel symbols, and
        since .so links permit undefined symbols that fails at import, not
        link time. Header absent / filter None = upstream all-combos build.
        """
        if dtype_filter is None:
            # No pruning: define every combo so all dispatch branches in
            # gemm_a8w8.cu compile to the real kernels (upstream behavior).
            combos = [
                (ABDtype, DDtype, EDtype)
                for EDtype in ["B16", "F16"]
                for ABDtype in ["I8", "F8"]
                for DDtype in ["F32", EDtype]
            ]
        else:
            combos = sorted(dtype_filter)
        lines = [
            "// generated by gen_instances.py -- do not edit",
            *(["#define AITER_PRUNE_DTYPES 1"] if dtype_filter is not None else []),
        ]
        for ab, d, e in combos:
            lines.append(f"#define AITER_BUILT_{ab}_{d}_{e} 1")
        content = "\n".join(lines) + "\n"
        Path(
            os.path.join(self.working_path, "gemm_a8w8_built_combos.h")
        ).write_text(content)

    def gen_instances(self, kernels_dict):
        if os.path.exists(self.impl_path):
            shutil.rmtree(self.impl_path)
        os.mkdir(self.impl_path)
        if os.path.exists(self.instances_path):
            shutil.rmtree(self.instances_path)
        os.mkdir(self.instances_path)

        kernels_filter, dtype_filter = _instance_filters(self.istune)
        if kernels_filter is not None:
            allowed_names = {kernels_list[i].name for i in kernels_filter}
            if self.istune:
                kept = {k: v for k, v in kernels_dict.items() if k in kernels_filter}
            else:
                # Non-tune dicts mix negative-int keys (default fallback
                # kernels, always kept -- the C++ heuristic dispatches into
                # them) and (gfx, cu_num, M, N, K) tuned-row keys (kept only
                # when their kernel is allowlisted; both the instance and its
                # lookup entry vanish together, so a dropped row falls back
                # to the heuristic, never to a missing symbol).
                kept = {
                    k: v
                    for k, v in kernels_dict.items()
                    if (isinstance(k, int) and k < 0) or v.name in allowed_names
                }
            dropped = len(kernels_dict) - len(kept)
            print(
                f"[aiter] AITER_CK_INSTANCE_LIST: keeping {len(kept)}/{len(kernels_dict)} "
                f"dict entries ({dropped} dropped)"
            )
            kernels_dict = kept
        self.dtype_filter = dtype_filter

        for k in kernels_dict.values():
            self.gen_instance(k)

        self.gen_built_combos_header(dtype_filter)
        self.gen_lookup_dict(kernels_dict)
        self.gen_manifest_head(kernels_dict)


def get_tune_dict(tune_dict_csv):
    if os.path.exists(tune_dict_csv):
        rows = pd.read_csv(tune_dict_csv)
        # The split-K sweep measures partition counts with the dispatched
        # kernel. Its placeholder kernelId=0 is not a measured tile choice.
        # The runtime still reads these rows for their splitK.
        if "kernelName" in rows:
            rows = rows[rows["kernelName"] != "ck_splitk_tuned"]
        return build_tune_dict(rows, default_kernels_dict, kernels_list)
    return default_kernels_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm a8w8 kernel",
    )

    # the directory for list_blobs/gen_blobs to write files into
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated",
    )

    parser.add_argument(
        "-f",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_gemm.csv",
        required=False,
        help="tune_file include the result after run gemm_a8w8_tune.py",
    )

    parser.add_argument(
        "--tune", action="store_true", required=False, help="generated tune instances"
    )

    # parser.add_argument(
    #     "--out_type",
    #     default="all",
    #     required=False,
    #     help="Specifie the type of scale\n \
    #         all: [bf16, fp16] \n  \
    #         bf16, fp16"
    # )

    # parser.add_argument(
    #     "--scale_type",
    #     default="all",
    #     required=False,
    #     help="Specifie the type of scale\n \
    #         all: [fp32, same as out] \n  \
    #         same: [same as out]"
    # )

    args = parser.parse_args()
    codegen = gemm_a8w8_fwd_codegen(args.working_path, args.tune)

    if args.tune:
        codegen.gen_instances(kernels_list)
    else:
        codegen.gen_instances(get_tune_dict(args.tune_file))
