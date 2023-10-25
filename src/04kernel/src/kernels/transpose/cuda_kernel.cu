#include "cuda_kernel.hh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace refactor::kernel {
    using namespace runtime;
    using Dim = TransposeInfo::Dimension;

    struct KernelFunctor {
        uint8_t const *data;
        uint8_t *transposed;
        Dim const *dims;
        long rank, eleSize;

        __device__ void operator()(long i) const noexcept {
            auto j = 0l, rem = i;
            for (auto k = 0l; k < rank; ++k) {
                auto const &d = dims[k];
                j += rem / d.strideO * d.strideI;
                rem %= d.strideO;
            }

            memcpy(transposed + i * eleSize, data + j * eleSize, eleSize);
        }
    };

    auto TransposeCuda::lower() const noexcept -> Routine {
        return [dims = thrust::device_vector<Dim>(info.dims.begin(), info.dims.end()),
                eleSize = static_cast<long>(dataType.size()),
                n = static_cast<long>(info.size)](
                   Resources &res, void const **inputs, void **outputs) {
            thrust::for_each_n(thrust::device,
                               thrust::counting_iterator<long>(0), n,
                               KernelFunctor{
                                   reinterpret_cast<uint8_t const *>(inputs[0]),
                                   reinterpret_cast<uint8_t *>(outputs[0]),
                                   dims.data().get(),
                                   static_cast<long>(dims.size()),
                                   eleSize,
                               });
        };
    }

}// namespace refactor::kernel
