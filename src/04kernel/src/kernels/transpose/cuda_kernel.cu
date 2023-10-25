#include "../../utilities/cuda/cuda_mem.cuh"
#include "cuda_kernel.hh"
#include <cuda.h>

namespace refactor::kernel {
    using namespace runtime;
    using Allocator = cuda::BasicCudaMemManager;

    struct Dimension {
        long strideI, strideO;
    };

    struct Helper {
        Dimension *dims;
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
        Helper helper;

        HelperContainer(TransposeInfo const &info, DataType dataType)
            : helper{
                  nullptr,
                  info.dims.size(),
                  dataType.size(),
                  info.size,
              } {
            auto bytes = helper.rank * sizeof(Dimension);
            helper.dims = reinterpret_cast<Dimension *>(Allocator::instance()->malloc(bytes));
            Allocator::instance()->copyHD(helper.dims, info.dims.data(), bytes);
        }

        ~HelperContainer() {
            Allocator::instance()->free(helper.dims);
        }
    };

    __global__ void transposeKernel(Helper helper, uint8_t const *data, uint8_t *transposed) {
        for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < helper.n; i += gridDim.x * blockDim.x) {
            memcpy(transposed + i * helper.eleSize, data + helper.locate(i) * helper.eleSize, helper.eleSize);
        }
    }

    auto TransposeCuda::lower() const noexcept -> Routine {
        return [container = std::make_shared<HelperContainer>(info, dataType)](
                   Resources &res, void const **inputs, void **outputs) {
            auto data = reinterpret_cast<uint8_t const *>(inputs[0]);
            auto transposed = reinterpret_cast<uint8_t *>(outputs[0]);
            constexpr static size_t blocksize = 1024;
            auto gridsize = (container->helper.n + blocksize - 1) / blocksize;
            transposeKernel<<<gridsize, blocksize>>>(container->helper, data, transposed);
        };
    }

}// namespace refactor::kernel
