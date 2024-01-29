#ifndef KERNEL_SELECT_H
#define KERNEL_SELECT_H

#include "../collector.h"

namespace refactor::kernel {

    enum class SelectType {
        Max,
        Min,
    };

    std::string_view opName(SelectType type);

    struct SelectCollector final : public InfoCollector {
        SelectType selectType;

        SelectCollector(decltype(_target), SelectType) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SELECT_H
