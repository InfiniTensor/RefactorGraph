#ifndef COMPUTATION_SLICE_H
#define COMPUTATION_SLICE_H

#include "../operator.h"
#include <vector>

namespace refactor::computation {

    struct Dim {
        int64_t start, step, number;
    };

    struct Slice final : public LayoutDependentOperator {
        std::vector<Dim> dims;

        explicit Slice(std::vector<Dim> dims_) noexcept
            : LayoutDependentOperator(),
              dims(std::move(dims_)) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SLICE_H
