#include "cuda_kernel.hh"
#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace refactor::kernel {
    using namespace runtime;
    using Dim = TransposeInfo::Dimension;

    struct Helper {
        Dim const *dims;
        long rank, eleSize, n;

        __device__ long locate(long rem) const noexcept {
            long ans = 0;
            for (long i = 0; i < rank; ++i) {
                ans += rem / dims[i].strideO * dims[i].strideI;
                rem %= dims[i].strideO;
            }
            return ans;
        };
    };

    struct HelperContainer {
        thrust::device_vector<Dim> dims;
        long eleSize, n;

        HelperContainer(TransposeInfo const &info, DataType dataType)
            : dims(info.dims.begin(), info.dims.end()),
              eleSize(dataType.size()),
              n(info.size) {
        }

        Helper helper() const {
            return {
                dims.data().get(),
                dims.size(),
                eleSize,
                n,
            };
        }
    };

    __global__ void transposeKernel(Helper helper, uint8_t const *data, uint8_t *transposed) {
        for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < helper.n; i += gridDim.x * blockDim.x) {
            memcpy(transposed + i * helper.eleSize, data + helper.locate(i) * helper.eleSize, helper.eleSize);
        }
    }

    auto TransposeCuda::lower() const noexcept -> Routine {
        return [container = HelperContainer(info, dataType)](
                   Resources &res, void const **inputs, void **outputs) {
            auto data = reinterpret_cast<uint8_t const *>(inputs[0]);
            auto transposed = reinterpret_cast<uint8_t *>(outputs[0]);
            constexpr static size_t blocksize = 1024;
            auto gridsize = (container.n + blocksize - 1) / blocksize;
            transposeKernel<<<gridsize, blocksize>>>(container.helper(), data, transposed);
        };
    }

}// namespace refactor::kernel
