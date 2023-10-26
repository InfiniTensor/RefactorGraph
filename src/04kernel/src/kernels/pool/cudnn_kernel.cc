#include "cudnn_kernel.hh"
#include "common.h"

namespace refactor::kernel {
    using K = PoolCudnn;

    K::PoolCudnn(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(PoolType poolType,
                  bool ceil,
                  KernelShape const &kernelShape,
                  PoolAttributes const &poolAttributes,
                  Tensor const &x,
                  Tensor const &y) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        // TODO check data type
        auto pb = poolAttributes.padsBegin(),
             pe = poolAttributes.padsEnd(),
             d = poolAttributes.dilations(),
             s = poolAttributes.strides();
        if (x.rank() != 4 ||
            d[0] != 1 ||
            d[1] != 1 ||
            pb[0] != pe[0] ||
            pb[1] != pe[1]) {
            return nullptr;
        }
        return std::make_unique<K>(decltype(info){
            poolType,
            x.dataType,
            {
                static_cast<int>(x.shape[0]),
                static_cast<int>(x.shape[1]),
                static_cast<int>(x.shape[2]),
                static_cast<int>(x.shape[3]),
            },
            {
                static_cast<int>(y.shape[0]),
                static_cast<int>(y.shape[1]),
                static_cast<int>(y.shape[2]),
                static_cast<int>(y.shape[3]),
            },
            {
                static_cast<int>(kernelShape[0]),
                static_cast<int>(kernelShape[1]),
            },
            {pb[0], pb[1]},
            {s[0], s[1]},
        });
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing pool using CUDNN";
    }

}// namespace refactor::kernel
