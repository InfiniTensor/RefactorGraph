#include "frontend/operator.h"
#include "frontend/graph.h"
#include "common.h"

namespace refactor::frontend {

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
                    UNREACHABLE();
            }
#undef CASE
        }
    }
    bool Attribute::operator!=(Attribute const &rhs) const {
        return !operator==(rhs);
    }

#define CHECK(TYPE, NAME)                           \
    bool Attribute::NAME() const {                  \
        return std::holds_alternative<TYPE>(value); \
    }

    CHECK(Int, isInt)
    CHECK(Ints, isInts)
    CHECK(Float, isFloat)
    CHECK(Floats, isFloats)
    CHECK(String, isString)
    CHECK(Strings, isStrings)
    CHECK(Tensor_, isTensor)
    CHECK(Tensors, isTensors)
#undef CHECK

#define CONVERT(TYPE, NAME)                        \
    TYPE &Attribute::NAME() {                      \
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

    static std::unordered_map<std::string, Operator::Builder> OP_BUILDERS;

    void Operator::register_(std::string opType, Builder builder) {
        auto [it, ok] = OP_BUILDERS.try_emplace(std::move(opType), builder);
        ASSERT(ok || it->second == builder, "Duplicate operator type");
    }
    OpBox Operator::build(std::string opType, Attributes attributes) {
        if (auto it = OP_BUILDERS.find(opType); it != OP_BUILDERS.end()) {
            return it->second(opType, std::move(attributes));
        }
        RUNTIME_ERROR(fmt::format("Unknown operator \"{}\"", opType));
    }

}// namespace refactor::frontend
