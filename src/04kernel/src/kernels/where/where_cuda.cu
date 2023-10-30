#include "where_cuda.hh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace refactor::kernel {
    using namespace runtime;

    // NOTICE NVCC 编译的对象无论是否在相同编译单元，都不可以重名。
    struct WhereKernelFunctor {
        bool const *c;
        uint8_t const *x;
        uint8_t const *y;
        uint8_t *output;
        uint_lv2 const *strides;
        long rank, eleSize;

        __device__ void operator()(long i) const noexcept {
            auto ic = 0l, ix = 0l, iy = 0l, rem = i;
            for (auto j = 0l; j < rank; ++j) {
                auto dim = strides + 4 * i;
                auto quot = rem / dim[3];
                rem %= dim[3];
                ic += quot * dim[0];
                ix += quot * dim[1];
                iy += quot * dim[2];
            }
            memcpy(output + i * eleSize,
                   c[ic]
                       ? x + ix * eleSize
                       : y + iy * eleSize,
                   eleSize);
        }
    };

    auto WhereCuda::lower() const noexcept -> Routine {
        return [strides = thrust::device_vector<uint_lv2>(broadcaster.strides.begin(), broadcaster.strides.end()),
                n = static_cast<long>(broadcaster.outputsCount),
                eleSize = static_cast<long>(dataType.size())](Resources &res, void const **inputs, void **outputs) {
            thrust::for_each_n(
                thrust::device, thrust::counting_iterator<long>(0), n,
                WhereKernelFunctor{
                    reinterpret_cast<bool const *>(inputs[0]),
                    reinterpret_cast<uint8_t const *>(inputs[1]),
                    reinterpret_cast<uint8_t const *>(inputs[2]),
                    reinterpret_cast<uint8_t *>(outputs[0]),
                    strides.data().get(),
                    static_cast<long>(strides.size() / 4),
                    eleSize,
                });
        };
    }

}// namespace refactor::kernel
