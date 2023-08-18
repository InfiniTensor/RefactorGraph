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
    using Attribute = std::variant<Int, Ints, Float, Floats, String, Strings>;
    using Attributes = std::unordered_map<std::string, Attribute>;

    struct NodeInfo {
        common::OpType opType;
        Attributes attributes;
    };
}// namespace refactor::graph

#endif// NODE_INFO_H
