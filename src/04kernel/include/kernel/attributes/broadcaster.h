#ifndef KERNEL_BROADCASTER_H
#define KERNEL_BROADCASTER_H

#include "../tensor.h"
#include <vector>

namespace refactor::kernel {

    using ShapeRefs = absl::InlinedVector<std::reference_wrapper<Shape const>, 2>;

    class Broadcaster {
        using ItRev = Shape::const_reverse_iterator;

        struct Input {
            ItRev it, end;
        };

        static std::vector<Input> build(TensorRefs const &);
        static std::vector<Input> build(ShapeRefs const &);

        explicit Broadcaster(std::vector<Input>) noexcept;

    public:
        std::vector<uint_lv2> strides;
        uint_lv2 outputsCount, inputsCount;

        explicit Broadcaster(TensorRefs const &inputs) noexcept;
        explicit Broadcaster(ShapeRefs const &inputs) noexcept;
        void locate(uint_lv2 k, uint_lv2 ans[]) const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_BROADCASTER_H
