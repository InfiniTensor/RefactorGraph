#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "kernel/cuda/gather.cuh"
#endif

namespace refactor::kernel {
    using K = GatherCuda;

    K::GatherCuda(GatherInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(GatherInfo info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return std::make_unique<K>(std::move(info));
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing gather using CUDA";
    }

#ifdef USE_CUDA
    auto K::lower(Resources &) const noexcept -> Routine {
        auto params = cuda::ThreadsDistributer()(info.prefix * info.midSizeO);
        return [info = this->info, params](Resources &, void const **inputs, void **outputs) {
            cuda::launchGather(
                params,
                inputs[0],
                inputs[1],
                outputs[0],
                info.idxType == DataType::I64,
                info.postfix,
                info.midSizeI,
                info.midSizeO);
        };
    }
#endif

}// namespace refactor::kernel
