#include "where_cuda.hh"

namespace refactor::kernel {
    using K = WhereCuda;

    K::WhereCuda(DataType dataType_, WhereBroadcast b) noexcept
        : Kernel(), dataType(dataType_), info(std::move(b)) {}
    auto K::build(Tensor const &c, Tensor const &x, Tensor const &y, Tensor const &o) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        return std::make_unique<K>(x.dataType, WhereBroadcast(c.shape, x.shape, y.shape, o.shape));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing where operation using CUDA";
    }

}// namespace refactor::kernel