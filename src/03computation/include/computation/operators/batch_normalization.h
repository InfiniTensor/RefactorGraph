#ifndef COMPUTATION_BATCH_NORMALIZATION_H
#define COMPUTATION_BATCH_NORMALIZATION_H

#include "../operator.h"

namespace refactor::computation {

    struct BatchNormalization final : public Operator {
        constexpr BatchNormalization() noexcept : Operator() {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_BATCH_NORMALIZATION_H
