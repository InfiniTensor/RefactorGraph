#include "cuda_kernel.hh"

namespace refactor::kernel {
    using namespace runtime;

    auto SplitCuda::lower() const noexcept -> Routine {
        // return [dims = thrust::device_vector<Dim>(info.dims.begin(), info.dims.end()),
        //         eleSize = static_cast<long>(dataType.size()),
        //         n = static_cast<long>(info.size)](
        //            Resources &res, void const **inputs, void **outputs) {
        //     thrust::for_each_n(thrust::device,
        //                        thrust::counting_iterator<long>(0), n,
        //                        KernelFunctor{
        //                            reinterpret_cast<uint8_t const *>(inputs[0]),
        //                            reinterpret_cast<uint8_t *>(outputs[0]),
        //                            dims.data().get(),
        //                            static_cast<long>(dims.size()),
        //                            eleSize,
        //                        });
        // };
    }

}// namespace refactor::kernel
