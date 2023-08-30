#include "graph/node_info.h"
#include "common/error_handler.h"

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
    TYPE Attribute::NAME() const {                 \
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
    CONVERT(String, string_)
    CONVERT(Strings, strings)
#undef CONVERT

    bool NodeInfo::operator==(NodeInfo const &rhs) const {
        return opType == rhs.opType && attributes == rhs.attributes;
    }
    bool NodeInfo::operator!=(NodeInfo const &rhs) const {
        return !operator==(rhs);
    }

}// namespace refactor::graph
