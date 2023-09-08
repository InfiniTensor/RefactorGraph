#include "graph/node_info.h"
#include "common/error_handler.h"
#include "graph/graph.h"

namespace refactor::graph {

    bool Attribute::operator==(Attribute const &rhs) const {
        if (value.index() != rhs.value.index()) {
            return false;
        } else {
#define CASE(I) \
    case I:     \
        return std::get<I>(value) == std::get<I>(rhs.value)
            switch (value.index()) {
                CASE(0);
                CASE(1);
                CASE(2);
                CASE(3);
                CASE(4);
                CASE(5);
                CASE(6);
                CASE(7);
                default:
                    RUNTIME_ERROR("Unreachable");
            }
#undef CASE
        }
    }
    bool Attribute::operator!=(Attribute const &rhs) const {
        return !operator==(rhs);
    }

#define CONVERT(TYPE, NAME)                        \
    TYPE const &Attribute::NAME() const {          \
        if (std::holds_alternative<TYPE>(value)) { \
            return std::get<TYPE>(value);          \
        } else {                                   \
            RUNTIME_ERROR("Attribute type error"); \
        }                                          \
    }

    CONVERT(Int, int_)
    CONVERT(Ints, ints)
    CONVERT(Float, float_)
    CONVERT(Floats, floats)
    CONVERT(String, string)
    CONVERT(Strings, strings)
    CONVERT(Tensor_, tensor)
    CONVERT(Tensors, tensors)
#undef CONVERT

    bool Operator::operator==(Operator const &rhs) const {
        return opType == rhs.opType && attributes == rhs.attributes;
    }
    bool Operator::operator!=(Operator const &rhs) const {
        return !operator==(rhs);
    }

    Attribute const &Operator::attribute(const char *name) const {
        return attributes.at(name);
    }

    Attribute const &Operator::attribute(const char *name, Attribute const &default_) const {
        if (auto it = attributes.find(name); it != attributes.end()) {
            return it->second;
        } else {
            return default_;
        }
    }

    bool Subgraph::operator==(Subgraph const &rhs) const {
        return false;
    }
    bool Subgraph::operator!=(Subgraph const &rhs) const {
        return !operator==(rhs);
    }

    NodeInfo::NodeInfo(Operator &&op) : info(std::forward<Operator>(op)) {}
    NodeInfo::NodeInfo(Subgraph &&sg) : info(std::forward<Subgraph>(sg)) {}

    bool NodeInfo::isOperator() const { return std::holds_alternative<Operator>(info); }
    bool NodeInfo::isSubgraph() const { return std::holds_alternative<Subgraph>(info); }

    Operator &NodeInfo::operator_() { return std::get<Operator>(info); }
    Operator const &NodeInfo::operator_() const { return std::get<Operator>(info); }
    Subgraph &NodeInfo::subgraph() { return std::get<Subgraph>(info); }
    Subgraph const &NodeInfo::subgraph() const { return std::get<Subgraph>(info); }

    bool NodeInfo::operator==(NodeInfo const &rhs) const {
        if (info.index() != rhs.info.index()) {
            return false;
        } else {
            switch (info.index()) {
                case 0:
                    return std::get<0>(info) == std::get<0>(rhs.info);
                case 1:
                    return std::get<1>(info) == std::get<1>(rhs.info);
                default:
                    RUNTIME_ERROR("Unreachable");
            }
        }
    }
    bool NodeInfo::operator!=(NodeInfo const &rhs) const {
        return !operator==(rhs);
    }

}// namespace refactor::graph
