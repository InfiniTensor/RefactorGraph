#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "kernel/cuda/rope.cuh"
#endif
namespace refactor::kernel {
    using K = RoPECuda;

    K::RoPECuda(decltype(info) info_, DataType _dtype) noexcept
        : Kernel(), info(info_), dtype(_dtype) {}

    auto K::build(decltype(info) info, Tensor const &x) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        return std::make_unique<K>(info, x.dataType);
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing rotary position embedding on Nvidia gpu";
    }
#ifdef USE_CUDA
    auto K::lower(Resources &) const -> RoutineWorkspace {

        return [info = this->info, useHalf = this->dtype == DataType::FP16]//
            (Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
                cuda::launchRoPE(
                    inputs[0],
                    reinterpret_cast<const int64_t *>(inputs[1]),
                    outputs[0],
                    info.batchsize,
                    info.seq_len,
                    info.n_heads,
                    info.head_dim,
                    info.theta,
                    useHalf);
            };
    }
#endif

}// namespace refactor::kernel
