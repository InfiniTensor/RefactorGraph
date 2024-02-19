#ifndef KERNEL_PAD_ATTRIBUTES_H
#define KERNEL_PAD_ATTRIBUTES_H

#include "../tensor.h"
#include "common.h"

namespace refactor::kernel {

    struct PadType {
        enum : uint8_t {
            Constant,
            Reflect,
            Edge,
            Wrap,
        } type;

        constexpr PadType() noexcept
            : type(Constant) {}
        constexpr PadType(decltype(type) type_) noexcept
            : type(type_) {}
        constexpr operator decltype(type)() const noexcept {
            return type;
        }
        constexpr std::string_view toString() const noexcept {
            switch (type) {
                case Constant:
                    return "Constant";
                case Reflect:
                    return "Reflect";
                case Edge:
                    return "Edge";
                case Wrap:
                    return "Wrap";
                default:
                    UNREACHABLE();
            }
        }
    };

    namespace pad {
        struct Dim {
            int64_t dimI, dimO, pads;
        };
    }// namespace pad

    using PadDimension = std::vector<pad::Dim>;

    struct PadInfo {
        struct Dim {
            dim_t strideI, strideO, padS, dimI;
        };
        std::vector<Dim> dims;
        dim_t blockCount, blockSize;

        PadInfo(decltype(dims), dim_t, dim_t) noexcept;
        PadInfo(PadDimension, Tensor const &);
        void reform(dim_t) noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_PAD_ATTRIBUTES_H
