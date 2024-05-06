#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "kernel/cuda/threads_distributer.cuh"
#include "kernel/cuda/topk.cuh"
#include <cuda_runtime.h>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#endif

namespace refactor::kernel {
    using K = TopKCuda;

    K::TopKCuda(TopKInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(TopKInfo info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return std::make_unique<K>(std::move(info));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing concat operation using CUDA";
    }

#ifdef USE_CUDA
    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        //return [info = this->info](Resources &, void *workspace, void const *const *inputs, void *const *outputs){

        //}
        return [info = this->info, params = cuda::ThreadsDistributer()(info.size.except_axis)]
            (Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            cuda::launchTopK(
                params,
                reinterpret_cast<float const *>(inputs[0]),
                reinterpret_cast<float *>(outputs[0]),
                reinterpret_cast<uint32_t *>(outputs[1]),
                info.topk,
                info.stride.axis,
                info.stride.in_pre,
                info.stride.out_pre, 
                info.size.axis);
        };
    }
#endif
}// namespace refactor::kernel
