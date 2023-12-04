#ifndef FRONTEND_TENSOR_H
#define FRONTEND_TENSOR_H

#include "graph_topo.h"
#include "kernel/blob.hh"
#include <span>
#include <unordered_set>
#include <variant>

namespace refactor::frontend {

    struct DimVariableInternal {
        std::string name;
        std::optional<int64_t> value;

        explicit DimVariableInternal(
            std::string,
            std::optional<int64_t> = std::nullopt);
    };

    using DimVariable = Arc<DimVariableInternal>;

    struct DimExpr {
        std::variant<int64_t, DimVariable> expr;

        bool operator==(DimExpr const &) const;
        bool operator!=(DimExpr const &) const;

        explicit DimExpr(int64_t);
        explicit DimExpr(std::string);
        explicit DimExpr(DimVariable);
        bool isValue() const;
        bool isVariable() const;
        bool hasValue() const;
        int64_t value() const;
        DimVariable const &variable() const;
    };

    using Shape = absl::InlinedVector<DimExpr, 4>;
    using ShapeSnapshot = absl::InlinedVector<std::variant<int64_t, DimVariable>, 4>;

    template<size_t N>
    using SmallInts = absl::InlinedVector<int64_t, N>;

    std::string shapeFormat(Shape const &);

    struct Tensor;

    struct TensorSnapshot {
        DataType dataType;
        ShapeSnapshot shape;
        std::weak_ptr<kernel::Blob> dataPtr;

        bool operator==(TensorSnapshot const &) const;
        bool operator!=(TensorSnapshot const &) const;
        bool operator==(Tensor const &) const;
        bool operator!=(Tensor const &) const;
    };

    /// @brief 张量边。
    struct Tensor {
        DataType dataType;
        Shape shape;
        Arc<kernel::Blob> data;

        std::unordered_set<DimVariable> depVariables;

        Tensor(DataType,
               Shape,
               Arc<kernel::Blob>,
               std::unordered_set<DimVariable>);
        static Arc<Tensor> share(Tensor const &);
        static Arc<Tensor> share(
            DataType,
            Shape,
            std::unordered_set<DimVariable>,
            Arc<kernel::Blob> = nullptr);

        int64_t rank() const;
        size_t elementsSize() const;
        size_t bytesSize() const;

        void *malloc();
        void free();

        TensorSnapshot snapshot() const;
    };

    using Tensor_ = Arc<Tensor>;

    struct Edge {
        Tensor_ tensor;
        std::string name;
    };

    class TensorRefs {
        std::vector<Edge> const &_edges;
        std::span<count_t const> _slice;

    public:
        TensorRefs(decltype(_edges) const &, decltype(_slice));
        Tensor const &operator[](size_t) const;
        size_t size() const;
        bool empty() const;

        class Iterator : public std::iterator<std::input_iterator_tag, std::reference_wrapper<Tensor const>> {
            TensorRefs const &_internal;
            size_t _index;

        public:
            Iterator(TensorRefs const &, size_t);
            bool operator==(Iterator const &) const;
            bool operator!=(Iterator const &) const;
            Tensor const &operator*() const;
            Iterator &operator++();
        };

        Iterator begin() const;
        Iterator end() const;
    };

}// namespace refactor::frontend

#endif// FRONTEND_TENSOR_H
