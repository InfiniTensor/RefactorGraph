#ifndef COMPUTATION_TENSOR_H
#define COMPUTATION_TENSOR_H

#include "absl/container/inlined_vector.h"
#include "common/data_type.h"
#include <cstring>
#include <variant>

namespace refactor::computation {

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

    /// @brief 内存块。
    struct Blob {
        /// @brief ! NOTICE 指针必须非空。
        void *ptr;

        explicit Blob(void *);
        Blob(Blob const &) = delete;
        Blob(Blob &&) = delete;

        ~Blob();
    };

    /// @brief 张量边。
    struct Tensor {
        common::DataType dataType;
        Shape shape;
        std::shared_ptr<Blob> data;

        Tensor(common::DataType, Shape, std::shared_ptr<Blob> = nullptr);

        bool operator==(Tensor const &) const;
        bool operator!=(Tensor const &) const;

        bool hasData() const;
        size_t elementsSize() const;
        size_t bytesSize() const;
    };

    using Edge = std::shared_ptr<Tensor>;

}// namespace refactor::computation

#endif// COMPUTATION_TENSOR_H
