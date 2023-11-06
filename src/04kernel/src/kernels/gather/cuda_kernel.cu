#include "cuda_kernel.hh"
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace refactor::kernel {
    using namespace runtime;

    // NOTICE NVCC 编译的对象无论是否在相同编译单元，都不可以重名。
    struct GatherKernelFunctor {
        uint8_t const *data, *indices;
        uint8_t *output;
        uint_lv2 postfix, midSizeI, midSizeO;
        bool i64;

        __device__ void operator()(long i) const noexcept {
            int64_t k = i64 ? reinterpret_cast<int64_t const *>(indices)[i % midSizeO]
                            : reinterpret_cast<int32_t const *>(indices)[i % midSizeO];
            memcpy(postfix * i + output,
                   postfix * (i / midSizeO * midSizeI + k) + data,// NOTICE 先除后乘不能反，因为除要向下取整
                   postfix);
        }
    };

    auto GatherCuda::lower(Resources &) const noexcept -> Routine {
        return [info = this->info](Resources &, void const **inputs, void **outputs) {
            thrust::for_each_n(thrust::device,
                               thrust::counting_iterator<long>(0), info.prefix * info.midSizeO,
                               GatherKernelFunctor{
                                   reinterpret_cast<uint8_t const *>(inputs[0]),
                                   reinterpret_cast<uint8_t const *>(inputs[1]),
                                   reinterpret_cast<uint8_t *>(outputs[0]),
                                   info.postfix,
                                   info.midSizeI,
                                   info.midSizeO,
                                   info.idxType == DataType::I64,
                               });
        };
    }

}// namespace refactor::kernel
