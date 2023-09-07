#ifndef EDGE_INFO_H
#define EDGE_INFO_H

#include "absl/container/inlined_vector.h"
#include "common/data_type.h"
#include <cstring>
#include <variant>

namespace refactor::graph {
    struct DimExpr {
        std::variant<int64_t, std::string> expr;

        bool operator==(DimExpr const &) const;
        bool operator!=(DimExpr const &) const;

        explicit DimExpr(int64_t);
        explicit DimExpr(std::string &&);
        bool isValue() const;
        bool isVariable() const;
        int64_t value() const;
        std::string const &variable() const;
    };

    using Shape = absl::InlinedVector<DimExpr, 4>;

    /// @brief 张量边。
    struct Tensor {
        common::DataType dataType;
        Shape shape;
        std::shared_ptr<uint8_t *> data;

        bool operator==(Tensor const &) const;
        bool operator!=(Tensor const &) const;
    };

}// namespace refactor::graph

#endif// EDGE_INFO_H
