#ifndef NODE_INFO_H
#define NODE_INFO_H

#include "absl/container/inlined_vector.h"
#include "common/op_type.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace refactor::graph {
    class Tensor;

    using Int = long long;
    using Ints = std::vector<long long>;
    using Float = double;
    using Floats = std::vector<double>;
    using String = std::string;
    using Strings = std::vector<std::string>;
    using Tensor_ = std::shared_ptr<Tensor>;
    using Tensors = std::vector<std::shared_ptr<Tensor>>;

    struct Attribute {
        std::variant<Int, Ints, Float, Floats, String, Strings, Tensor_, Tensors> value;

        bool operator==(Attribute const &) const;
        bool operator!=(Attribute const &) const;

        Int const &int_() const;
        Ints const &ints() const;
        Float const &float_() const;
        Floats const &floats() const;
        String const &string() const;
        Strings const &strings() const;
        Tensor_ const &tensor() const;
        Tensors const &tensors() const;
    };
    using Attributes = std::unordered_map<std::string, Attribute>;

    struct Operator {
        common::OpType opType;
        Attributes attributes;

        bool operator==(Operator const &) const;
        bool operator!=(Operator const &) const;

        Attribute const &attribute(const char *) const;
        Attribute const &attribute(const char *, Attribute const &default_) const;
    };

    class GraphMut;

    struct Subgraph {
        std::shared_ptr<GraphMut> graph;

        bool operator==(Subgraph const &) const;
        bool operator!=(Subgraph const &) const;
    };

    struct NodeInfo {
        std::variant<Operator, Subgraph> info;

        NodeInfo(Operator &&);
        NodeInfo(Subgraph &&);

        bool isOperator() const;
        bool isSubgraph() const;

        Operator &operator_();
        Operator const &operator_() const;
        Subgraph &subgraph();
        Subgraph const &subgraph() const;

        bool operator==(NodeInfo const &) const;
        bool operator!=(NodeInfo const &) const;
    };
}// namespace refactor::graph

#endif// NODE_INFO_H
