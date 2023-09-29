#ifndef COMPUTATION_SOFTMAX_H
#define COMPUTATION_SOFTMAX_H

#include "../operator.h"

namespace refactor::computation {

    struct Softmax final : public Operator {
        size_t axis;

        constexpr explicit Softmax(size_t axis_)
            : Operator(), axis(axis_) {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SOFTMAX_H
