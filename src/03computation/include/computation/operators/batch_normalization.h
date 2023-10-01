#ifndef COMPUTATION_BATCH_NORMALIZATION_H
#define COMPUTATION_BATCH_NORMALIZATION_H

#include "../operator.h"

namespace refactor::computation {

    struct BatchNormalization final : public Operator {
        constexpr BatchNormalization() : Operator() {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_BATCH_NORMALIZATION_H
