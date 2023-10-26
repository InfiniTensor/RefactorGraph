#include "cuda_kernel.hh"


namespace refactor::kernel {
    using K = GatherCuda;

    K::GatherCuda(DataType indexType_, GatherMetaData metaData_) noexcept
        : Kernel(), indexType(indexType_), metaData(std::move(metaData_)) {}

    auto K::build(Tensor const &in, Tensor const &indices, Tensor const &out, uint32_t axis) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
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
        return "Performing gather using CUDA";
    }

}// namespace refactor::kernel
