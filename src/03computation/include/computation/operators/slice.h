#ifndef COMPUTATION_SLICE_H
#define COMPUTATION_SLICE_H

#include "../operator.h"
#include <vector>

namespace refactor::computation {

    struct Dim {
        int64_t start, step, number;
    };

    struct Slice final : public Operator {
        std::vector<Dim> dims;

        explicit Slice(std::vector<Dim> dims_)
            : Operator(), dims(std::move(dims_)) {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
        bool isLayoutDependent() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SLICE_H
