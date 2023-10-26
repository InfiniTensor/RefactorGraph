#include "cpu_kernel.hh"
#include "common.h"

namespace refactor::kernel {
    using K = GatherCpu;

    K::GatherCpu(DataType indexType_, GatherMetaData metaData_) noexcept
        : Kernel(), indexType(indexType_), metaData(std::move(metaData_)) {}

    auto K::build(Tensor const &in, Tensor const &indices, Tensor const &out, uint32_t axis) noexcept -> KernelBox {
        // index has to be int32/int64
        if (indices.dataType != DataType::I32 && indices.dataType != DataType::I64) {
            return nullptr;
        }

        GatherMetaData md;
        md.itemSize = in.dataType.size();
        md.outSize = out.elementsSize();
        md.axis = axis;
        md.inNDim = in.rank();
        md.outNDim = out.rank();
        md.idxNDim = indices.rank();
        md.idxShape = indices.shape;
        md.outShape = out.shape;
        md.inStrides = in.strides();
        md.idxStrides = indices.strides();

        return std::make_unique<K>(indices.dataType, md);
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing gather using CPU";
    }

    // Given the offset of gathered output, return the corresponding
    // offset of the input.
    template<typename Tind>
    size_t gatheredOffset2Offset(size_t gOffset, Tind *indices,
                                 GatherMetaData const *metaData) {
        size_t offset = 0;
        for (int i = metaData->inNDim - 1, k = metaData->outNDim - 1; i >= 0; --i) {
            size_t idx = 0;
            if (i == metaData->axis) {
                size_t idxOffset = 0;
                for (int j = metaData->idxNDim - 1; j >= 0; --j) {
                    size_t p = gOffset % metaData->idxShape[j];
                    gOffset = gOffset / metaData->idxShape[j];
                    idxOffset += p * metaData->idxStrides[j];
                }
                idx = indices[idxOffset];
                k = k - metaData->idxNDim;

            } else {
                idx = gOffset % metaData->outShape[k];
                gOffset = gOffset / metaData->outShape[k];
                --k;
            }
            offset += idx * metaData->inStrides[i];
        }
        return offset;
    }

    Routine K::lower() const noexcept {
        if (indexType == DataType::I64) {
            return [md = std::move(metaData)](runtime::Resources &, void const **inputs, void **outputs) {
                auto in = static_cast<char const *>(inputs[0]);
                auto indices = static_cast<size_t const *>(inputs[1]);
                auto out = static_cast<char *>(outputs[0]);
#pragma omp parallel for private(offset, md)
                for (size_t i = 0; i < md.outSize; i++) {
                    size_t offset = gatheredOffset2Offset(i, indices, &md);
                    memcpy(out + i * md.itemSize, in + offset * md.itemSize, md.itemSize);
                }
            };
        } else {// DataType::I32
            return [md = std::move(metaData)](runtime::Resources &, void const **inputs, void **outputs) {
                auto in = static_cast<char const *>(inputs[0]);
                auto indices = static_cast<int const *>(inputs[1]);
                auto out = static_cast<char *>(outputs[0]);
#pragma omp parallel for private(offset, md)
                for (size_t i = 0; i < md.outSize; i++) {
                    size_t offset = gatheredOffset2Offset(i, indices, &md);
                    memcpy(out + i * md.itemSize, in + offset * md.itemSize, md.itemSize);
                }
            };
        }
    }

}// namespace refactor::kernel
