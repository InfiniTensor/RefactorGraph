#include "where_cuda.hh"

namespace refactor::kernel {
    using K = WhereCuda;

    K::WhereCuda(DataType dataType_, Broadcaster b) noexcept
        : Kernel(),
          dataType(dataType_),
          broadcaster(std::move(b)) {}

    auto K::build(TensorRefs const &inputs) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        return std::make_unique<K>(inputs[1].get().dataType, Broadcaster(inputs));
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
