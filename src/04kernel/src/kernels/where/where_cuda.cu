#include "kernel/cuda/where.cuh"
#include "where_cuda.hh"
#include <thrust/device_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    auto WhereCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
        return [strides = thrust::device_vector<dim_t>(broadcaster.strides.begin(), broadcaster.strides.end()),
                params = cuda::ThreadsDistributer()(broadcaster.outputsCount),
                eleSize = static_cast<long>(dataType.size())](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
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
