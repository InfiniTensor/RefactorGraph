#include "cuda_kernel.hh"
#include "kernel/cuda/concat.cuh"
#include "mem_manager/foreign_blob.hh"
#include "runtime/mem_manager.hh"
#include <thrust/device_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    Routine ConcatCuda::lower(Resources &) const noexcept {
        auto sub = std::min(info.submultiple(), 16u);
        return [segments = thrust::device_vector<uint_lv2>(info.segments.begin(), info.segments.end()),
                params = cuda::ThreadsDistributer()(info.blockCount * info.sum / sub),
                sum = info.sum / sub,
                sub](Resources &res, void const **inputs, void **outputs) {
            auto size = segments.size() * sizeof(void *);
            auto inputs_ = mem_manager::ForeignBlob::share(res.fetch<MemManager>()->manager, size);
            inputs_->copyIn(inputs, size);
            cuda::launchConcat(
                params,
                reinterpret_cast<void const **>((void *) *inputs_),
                segments.data().get(),
                outputs[0],
                segments.size(),
                sum,
                sub);
        };
    }

}// namespace refactor::kernel
