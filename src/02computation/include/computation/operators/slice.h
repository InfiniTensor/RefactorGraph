#ifndef COMPUTATION_SLICE_H
#define COMPUTATION_SLICE_H

#include "../operator.h"
#include <vector>

namespace refactor::computation {

    struct Dim {
        int64_t start, step, number;
    };

    struct Slice : public Operator {
        std::vector<Dim> dims;

        explicit Slice(std::vector<Dim> dims_)
            : Operator(), dims(std::move(dims_)) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_SLICE_H
