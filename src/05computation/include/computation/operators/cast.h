#ifndef COMPUTATION_CAST_H
#define COMPUTATION_CAST_H

#include "../operator.h"
#include "common/data_type.h"

namespace refactor::computation {

    struct Cast final : public Operator {
        common::DataType targetDataType;

        constexpr explicit Cast(common::DataType targetDataType_) noexcept
            : Operator(), targetDataType(targetDataType_) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CAST_H
