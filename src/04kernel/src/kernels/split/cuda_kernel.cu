#include "cuda_kernel.hh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace refactor::kernel {
    using namespace runtime;

    struct KernelFunctor {
        void const *data;
        void **outputs;
        uint_lv2 const *segments;
        long outputCount, sum;

        __device__ void operator()(long i) const noexcept {
            auto offset = i * sum;
            for (auto j = 0; j < outputCount; ++j) {
                auto len = segments[j];
                auto out = reinterpret_cast<uint8_t *>(outputs[j]);
                // memcpy(out + i * len, data + offset, len);
                offset += len;
            }
        }
    };

    auto SplitCuda::lower() const noexcept -> Routine {
        return [segments = thrust::device_vector<uint_lv2>(info.segments.begin(), info.segments.end()),
                blockCount = info.blockCount,
                sum = info.sum](Resources &res, void const **inputs, void **outputs) {
            // thrust::for_each_n(thrust::device,
            //                    thrust::counting_iterator<long>(0), blockCount,
            //                    KernelFunctor{
            //                        inputs[0],
            //                        outputs,
            //                        segments.data().get(),
            //                        static_cast<long>(segments.size()),
            //                        static_cast<long>(sum),
            //                    });
        };
    }

}// namespace refactor::kernel
