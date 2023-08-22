#ifndef NODE_INFO_H
#define NODE_INFO_H

#include "common/op_type.h"
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace refactor::graph {
    using Int = long long;
    using Ints = std::vector<long long>;
    using Float = double;
    using Floats = std::vector<double>;
    using String = std::string;
    using Strings = std::vector<std::string>;

    struct Attribute {
        std::variant<Int, Ints, Float, Floats, String, Strings> value;

        bool operator==(Attribute const &) const;
        bool operator!=(Attribute const &) const;

        Int int_() const;
        Ints ints() const;
        Float float_() const;
        Floats floats() const;
        String string_() const;
        Strings strings() const;
    };
    using Attributes = std::unordered_map<std::string, Attribute>;

    struct NodeInfo {
        common::OpType opType;
        Attributes attributes;

        bool operator==(NodeInfo const &) const;
        bool operator!=(NodeInfo const &) const;
    };
}// namespace refactor::graph

#endif// NODE_INFO_H
