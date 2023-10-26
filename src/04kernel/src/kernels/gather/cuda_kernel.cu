#include "cuda_kernel.hh"
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace refactor::kernel {
    using namespace runtime;

    struct KernelFunctor {
        uint8_t const *data, *indices;
        uint8_t *output;
        uint_lv2 postfix, midSizeI, midSizeO;
        DataType idxType;

        __device__ void operator()(long i) const noexcept {
            for (long j = 0; j < midSizeO; ++j) {
                int64_t k = idxType.internal == DataType::I64
                                ? reinterpret_cast<int64_t const *>(indices)[j]
                                : reinterpret_cast<int32_t const *>(indices)[j];
                memcpy(postfix * (i * midSizeO + j) + output,
                       postfix * (i * midSizeI + k) + data,
                       postfix);
            }
        }
    };

    auto GatherCuda::lower() const noexcept -> Routine {
        return [info = this->info](Resources &, void const **inputs, void **outputs) {
            thrust::for_each_n(thrust::device,
                               thrust::counting_iterator<long>(0), info.prefix,
                               KernelFunctor{
                                   reinterpret_cast<uint8_t const *>(inputs[0]),
                                   reinterpret_cast<uint8_t const *>(inputs[1]),
                                   reinterpret_cast<uint8_t *>(outputs[0]),
                                   info.postfix,
                                   info.midSizeI,
                                   info.midSizeO,
                                   info.idxType,
                               });
        };
    }

}// namespace refactor::kernel
