#include "where_cuda.hh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace refactor::kernel {
    using namespace runtime;

    struct WhereKernelFunctor {
        bool const *c;
        uint8_t const *x;
        uint8_t const *y;
        uint8_t *output;
        long eleSize, rank;
        uint_lv2 const *strides;

        __device__ void operator()(long i) const noexcept {
            long ic = 0l, ix = 0l, iy = 0l;
            long rem = i;
            for (long j = 0l; j < 4; ++j) {
                long d = rem / strides[4 * j];
                ic += strides[4 * j + 1] * d;
                ix += strides[4 * j + 2] * d;
                iy += strides[4 * j + 3] * d;
                rem = rem % strides[4 * j];
            }
            memcpy(output + i * eleSize, *(c + ic) ? x + ix * eleSize : y + iy * eleSize, eleSize);
        }
    };

    auto WhereCuda::lower() const noexcept -> Routine {
        return [n = static_cast<long>(info._size),
                eleSize = static_cast<long>(dataType.size()),
                rank = static_cast<long>(info._strides.size() / 4),
                strides = thrust::device_vector<uint_lv2>(info._strides.begin(), info._strides.end())](Resources &res, void const **inputs, void **outputs) {
            thrust::for_each_n(
                thrust::device, thrust::counting_iterator<long>(0), n,
                WhereKernelFunctor{
                    reinterpret_cast<bool const *>(inputs[0]),
                    reinterpret_cast<uint8_t const *>(inputs[1]),
                    reinterpret_cast<uint8_t const *>(inputs[2]),
                    reinterpret_cast<uint8_t *>(outputs[0]),
                    eleSize,
                    rank,
                    strides.data().get(),
                });
        };
    }

}// namespace refactor::kernel
