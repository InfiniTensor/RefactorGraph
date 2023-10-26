#include "cuda_kernel.hh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    template<typename Tind>
    struct GatherFunction {
        uint8_t const *data;
        Tind const *indices;
        uint8_t *output;
        GatherCudaMetaData metaData;

        __device__ size_t gatheredOffset2Offset(size_t gOffset) const {
            size_t offset = 0;
            for (int i = metaData.inNDim - 1, k = metaData.outNDim - 1; i >= 0; --i) {
                size_t idx = 0;
                if (i == metaData.axis) {
                    size_t idxOffset = 0;
                    for (int j = metaData.idxNDim - 1; j >= 0; --j) {
                        size_t p = gOffset % metaData.idxShape[j];
                        gOffset = gOffset / metaData.idxShape[j];
                        idxOffset += p * metaData.idxStrides[j];
                    }
                    idx = indices[idxOffset];
                    k = k - metaData.idxNDim;
                } else {
                    idx = gOffset % metaData.outShape[k];
                    gOffset = gOffset / metaData.outShape[k];
                    --k;
                }
                offset += idx * metaData.inStrides[i];
            }
            return offset;
        }

        __device__ void operator()(size_t i) const noexcept {
            size_t offset = gatheredOffset2Offset(i);
            memcpy(output + i * metaData.itemSize, data + offset * metaData.itemSize, metaData.itemSize);
        }
    };

    auto
    GatherCuda::lower() const noexcept -> Routine {
        thrust::device_vector<uint32_t> idxShape(metaData.idxShape.begin(), metaData.idxShape.end());
        thrust::device_vector<uint32_t> outShape(metaData.outShape.begin(), metaData.outShape.end());
        thrust::device_vector<size_t> inStrides(metaData.inStrides.begin(), metaData.inStrides.end());
        thrust::device_vector<size_t> idxStrides(metaData.idxStrides.begin(), metaData.idxStrides.end());

        if (indexType == DataType::I32) {
            return [md_ = std::move(metaData),
                    idxShape_ = std::move(idxShape),
                    outShape_ = std::move(outShape),
                    inStrides_ = std::move(inStrides),
                    idxStrides_ = std::move(idxStrides)](Resources &res, void const **inputs, void **outputs) {
                auto in = static_cast<uint8_t const *>(inputs[0]);
                auto indices = static_cast<int const *>(inputs[1]);
                auto out = static_cast<uint8_t *>(outputs[0]);

                GatherCudaMetaData cudaMD;
                cudaMD.itemSize = md_.itemSize;
                cudaMD.outSize = md_.outSize;
                cudaMD.axis = md_.axis;
                cudaMD.inNDim = md_.inNDim;
                cudaMD.outNDim = md_.outNDim;
                cudaMD.idxNDim = md_.idxNDim;
                cudaMD.idxShape = idxShape_.data().get();
                cudaMD.outShape = outShape_.data().get();
                cudaMD.inStrides = inStrides_.data().get();
                cudaMD.idxStrides = idxStrides_.data().get();

                thrust::for_each_n(thrust::device,
                                   thrust::counting_iterator<size_t>(0), cudaMD.outSize,
                                   GatherFunction<int>{
                                       in,
                                       indices,
                                       out,
                                       cudaMD,
                                   });
            };
        } else {// DataType::I64
            return [md_ = std::move(metaData),
                    idxShape_ = std::move(idxShape),
                    outShape_ = std::move(outShape),
                    inStrides_ = std::move(inStrides),
                    idxStrides_ = std::move(idxStrides)](Resources &res, void const **inputs, void **outputs) {
                auto in = static_cast<uint8_t const *>(inputs[0]);
                auto indices = static_cast<int64_t const *>(inputs[1]);
                auto out = static_cast<uint8_t *>(outputs[0]);

                GatherCudaMetaData cudaMD;
                cudaMD.itemSize = md_.itemSize;
                cudaMD.outSize = md_.outSize;
                cudaMD.axis = md_.axis;
                cudaMD.inNDim = md_.inNDim;
                cudaMD.outNDim = md_.outNDim;
                cudaMD.idxNDim = md_.idxNDim;
                cudaMD.idxShape = idxShape_.data().get();
                cudaMD.outShape = outShape_.data().get();
                cudaMD.inStrides = inStrides_.data().get();
                cudaMD.idxStrides = idxStrides_.data().get();

                thrust::for_each_n(thrust::device,
                                   thrust::counting_iterator<size_t>(0), cudaMD.outSize,
                                   GatherFunction<int64_t>{
                                       in,
                                       indices,
                                       out,
                                       cudaMD,
                                   });
            };
        }
    }

}// namespace refactor::kernel
