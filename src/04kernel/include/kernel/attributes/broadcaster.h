#ifndef KERNEL_BROADCASTER_H
#define KERNEL_BROADCASTER_H

#include "../tensor.h"
#include <vector>

namespace refactor::kernel {

    class Broadcaster {
        static std::vector<slice_t<uint_lv2>> build(TensorRefs const &);

    public:
        std::vector<uint_lv2> strides;
        uint_lv2 outputsCount, inputsCount;

        explicit Broadcaster(std::vector<slice_t<uint_lv2>>) noexcept;
        explicit Broadcaster(TensorRefs const &inputs) noexcept;
        void locate(uint_lv2 k, uint_lv2 ans[]) const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_BROADCASTER_H
