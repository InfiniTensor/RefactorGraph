#include "cuda_kernel.hh"
#include "mem_manager/foreign_blob.hh"
#include "runtime/mem_manager.hh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace refactor::kernel {
    using namespace runtime;

    // NOTICE NVCC 编译的对象无论是否在相同编译单元，都不可以重名。
    struct SplitKernelFunctor {
        uint8_t const *data;
        uint8_t **outputs;
        uint_lv2 const *segments;
        long outputCount, sum;

        __device__ void operator()(long i) const noexcept {
            auto offset = i * sum;
            for (auto j = 0; j < outputCount; ++j) {
                auto len = segments[j];
                memcpy(outputs[j] + i * len, data + offset, len);
                offset += len;
            }
        }
    };

    Routine SplitCuda::lower(Resources &) const noexcept {
        return [segments = thrust::device_vector<uint_lv2>(info.segments.begin(), info.segments.end()),
                blockCount = info.blockCount,
                sum = info.sum](Resources &res, void const **inputs, void **outputs) {
            auto size = segments.size() * sizeof(void *);
            auto outputs_ = mem_manager::ForeignBlob::share(res.fetch<MemManager>()->manager, size);
            outputs_->copyIn(outputs, size);
            thrust::for_each_n(thrust::device,
                               thrust::counting_iterator<long>(0), blockCount,
                               SplitKernelFunctor{
                                   reinterpret_cast<uint8_t const *>(inputs[0]),
                                   reinterpret_cast<uint8_t **>((void *) *outputs_),
                                   segments.data().get(),
                                   static_cast<long>(segments.size()),
                                   static_cast<long>(sum),
                               });
        };
    }

}// namespace refactor::kernel
