#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "kernel/cuda/threads_distributer.cuh"
#endif

namespace refactor::kernel {
    using K = SelectCuda;

    K::SelectCuda(decltype(dataType) dataType_,
                  decltype(selectType) selectType_,
                  decltype(broadcaster) broadcaster_,
                  decltype(inputsNum) inputsNum_) noexcept
        : dataType(dataType_),
          selectType(selectType_),
          broadcaster(broadcaster_),
          inputsNum(inputsNum_) {}

    auto K::build(SelectType selectType_, TensorRefs inputs_) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return std::make_unique<K>(inputs_[0].get().dataType, selectType_, Broadcaster(inputs_), inputs_.size());
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing select operation on Nvidia GPU";
    }

#ifdef USE_CUDA

    constexpr static const char *NO_BROADCAST = R"~(
struct Inputs {{
    {dt} const *const addr[{inputsNum}];
}};

__device__ __forceinline__ static {dt} fn({dt} a, {dt} b) {{
    return {op};
}}

extern "C" __global__ void kernel(
    {dt} *__restrict__ output,
    Inputs inputs
) {{
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < {n};
         tid += step) {{
        output[tid] = inputs.addr[0][tid];
        for (auto idx = 1; idx < {inputsNum}; ++idx) {{
            output[tid] = fn(inputs.addr[idx][tid], output[tid]);
        }}
    }}
}}
)~";

    constexpr static const char *BROADCAST = R"~(
struct Inputs {{
    {dt} const *const addr[{inputsNum}];
}};

struct Strides {{
    unsigned int s[({inputsNum}+1) * {rank}];
}};

__device__ __forceinline__ static {dt} fn({dt} a, {dt} b) {{
    return {op};
}}

extern "C" __global__ void kernel(
    {dt} *__restrict__ output,
    Inputs inputs,
    Strides strides
) {{
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < {n};
         tid += step) {{
        auto rem = tid;
        size_t ans[{inputsNum}]{{}};
        for (auto i = 0; i < {rank}; ++i) {{
            auto dim = strides.s + ({inputsNum} + 1) * i;
            auto quot = rem / dim[{inputsNum}];
            for (auto j = 0; j < {inputsNum}; ++j) {{ ans[j] += dim[j] * quot; }}
            rem %= dim[{inputsNum}];
        }}
        output[tid] = inputs.addr[0][ans[0]];
        for (auto idx = 1; idx < {inputsNum}; ++idx) {{
            output[tid] = fn(inputs.addr[idx][ans[idx]], output[tid]);
        }}
    }}
}}
)~";

    constexpr static std::string_view op(SelectType op, DataType dt) {
        switch (op) {
            case SelectType::Max:
                return "a > b ? a : b";
            case SelectType::Min:
                return "a < b ? a : b";
            default:
                UNREACHABLE();
        }
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;

        auto postfix = fmt::format("_{}_{}", dataType.name(), opName(selectType));
        auto dt_ = nvrtc::dataType(dataType);
        auto op_ = op(selectType, dataType);
        auto params = cuda::ThreadsDistributer()(broadcaster.outputsCount);

        if (!broadcaster.needBroadcast()) {
            auto name = fmt::format("select{}", postfix);
            auto code = fmt::format(NO_BROADCAST,
                                    fmt::arg("dt", dt_),
                                    fmt::arg("op", op_),
                                    fmt::arg("inputsNum", inputsNum),
                                    fmt::arg("n", params.n));
            return [params, h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel")]//
                (Resources &, void *, void const *const *inputs, void *const *outputs) {
                    auto output = outputs[0];
                    void *args[]{&output, const_cast<void **>(inputs)};
                    h->launch(params.gridSize, 1, 1,
                              params.blockSize, 1, 1,
                              0, args);
                };
        } else {
            auto name = fmt::format("select{}", postfix);
            auto rank = broadcaster.strides.size() / (broadcaster.inputsCount + 1);
            auto code = fmt::format(
                BROADCAST,
                fmt::arg("dt", dt_),
                fmt::arg("op", op_),
                fmt::arg("inputsNum", inputsNum),
                fmt::arg("n", params.n),
                fmt::arg("rank", rank));
            return [params, h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                    strides = broadcaster.strides]//
                (Resources &, void *, void const *const *inputs, void *const *outputs) {
                    void *args[]{const_cast<void **>(outputs), const_cast<void **>(inputs), const_cast<dim_t *>(strides.data())};
                    h->launch(params.gridSize, 1, 1,
                              params.blockSize, 1, 1,
                              0, args);
                };
        }
    }

#endif

}// namespace refactor::kernel
