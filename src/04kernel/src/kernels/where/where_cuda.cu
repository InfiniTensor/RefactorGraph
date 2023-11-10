#include "kernel/cuda/where.cuh"
#include "where_cuda.hh"
#include <thrust/device_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    Routine WhereCuda::lower(Resources &) const noexcept {
        return [strides = thrust::device_vector<uint_lv2>(broadcaster.strides.begin(), broadcaster.strides.end()),
                params = cuda::ThreadsDistributer()(broadcaster.outputsCount),
                eleSize = static_cast<long>(dataType.size())](Resources &res, void const **inputs, void **outputs) {
            cuda::launchWhere(
                params,
                strides.data().get(),
                inputs[0],
                inputs[1],
                inputs[2],
                outputs[0],
                strides.size() / 4,
                eleSize);
        };
    }

}// namespace refactor::kernel
