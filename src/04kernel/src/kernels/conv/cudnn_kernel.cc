#include "cudnn_kernel.hh"

namespace refactor::kernel {
    using K = ConvCudnn;

    K::ConvCudnn(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(PoolAttributes const &poolAttributes,
                  Tensor const &x,
                  Tensor const &w,
                  std::optional<std::reference_wrapper<Tensor const>> b,
                  Tensor const &y) -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        std::optional<ExpandInfo> biasExpand = std::nullopt;
        if (b) {
            ASSERT(b->get().shape[0] == y.shape[1], "");
            std::vector<dim_t> input(y.rank(), 1);
            input[1] = y.shape[1];
            biasExpand.emplace(ExpandInfo(
                b->get().dataType,
                slice(input.data(), input.size()),
                slice(y.shape.data(), y.rank())));
        }

        // group is not supported
        if (w.rank() != 4 || x.shape[1] != w.shape[1]) {
            return nullptr;
        }
        auto padsBegin = poolAttributes.padsBegin(),
             padsEnd = poolAttributes.padsEnd();
        if (padsBegin[0] != padsEnd[0] ||
            padsBegin[1] != padsEnd[1]) {
            return nullptr;
        }
        auto d = poolAttributes.dilations(),
             p = poolAttributes.pads(),
             s = poolAttributes.strides();
        return std::make_unique<K>(decltype(info){
            x.dataType,
            {
                static_cast<int>(x.shape[0]),
                static_cast<int>(x.shape[1]),
                static_cast<int>(x.shape[2]),
                static_cast<int>(x.shape[3]),
            },
            {
                static_cast<int>(w.shape[0]),
                static_cast<int>(w.shape[1]),
                static_cast<int>(w.shape[2]),
                static_cast<int>(w.shape[3]),
            },
            {
                static_cast<int>(y.shape[0]),
                static_cast<int>(y.shape[1]),
                static_cast<int>(y.shape[2]),
                static_cast<int>(y.shape[3]),
            },
            {d[0], d[1]},
            {p[0], p[1]},
            {s[0], s[1]},
            std::move(biasExpand),
        });
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing conv using CUDNN";
    }

}// namespace refactor::kernel
