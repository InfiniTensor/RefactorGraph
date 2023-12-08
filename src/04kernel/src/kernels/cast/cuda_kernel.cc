#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "kernel/cuda/threads_distributer.cuh"
#include <cuda_runtime.h>
#endif

namespace refactor::kernel {
    using K = CastCuda;

    K::CastCuda(decltype(from) from_,
                decltype(to) to_,
                decltype(size) size_) noexcept
        : from(from_), to(to_), size(size_) {}

    auto K::build(Tensor const &from, Tensor const &to) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return std::make_unique<K>(from.dataType, to.dataType, from.elementsSize());
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing cast operation on Nvidia gpu";
    }

}// namespace refactor::kernel
