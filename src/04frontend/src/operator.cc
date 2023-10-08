#include "frontend/operator.h"
#include "common/error_handler.h"
#include "frontend/graph.h"

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

    bool OpType::operator==(OpType const &rhs) const { return id == rhs.id; }
    bool OpType::operator!=(OpType const &rhs) const { return !operator==(rhs); }

    struct OpProperties {
        InferFn inference;
        LowerFn lower;

        bool operator==(OpProperties const &rhs) const {
            return inference == rhs.inference && lower == rhs.lower;
        }
        bool operator!=(OpProperties const &rhs) const { return !operator==(rhs); }
    };

    struct Op {
        std::string_view name;
        OpProperties properties;
    };

    struct OpRepo {
        std::vector<Op> map;
        std::unordered_map<std::string, size_t> nameMap;
        std::unordered_map<std::string, OpProperties> knownList;
    } static OP_REPO;

    void OpType::register_(const char *name, InferFn infer, LowerFn lower) {
        OpProperties properties{infer, lower};
        std::string name_(name);
        if (auto it = OP_REPO.nameMap.find(name_); it != OP_REPO.nameMap.end()) {
            ASSERT(OP_REPO.map.at(it->second).properties == properties,
                   "Different inference function for the same operator");
        }
        if (auto it = OP_REPO.knownList.find(name_); it != OP_REPO.knownList.end()) {
            ASSERT(it->second == properties,
                   "Different inference function for the same operator");
        }
        OP_REPO.knownList.insert({std::move(name_), properties});
    }
    OpType OpType::parse(std::string name) {
        if (auto it = OP_REPO.nameMap.find(name); it != OP_REPO.nameMap.end()) {
            return {it->second};
        } else if (auto it = OP_REPO.knownList.find(name); it != OP_REPO.knownList.end()) {
            auto id = OP_REPO.map.size();
            auto [it_, ok] = OP_REPO.nameMap.insert({std::move(name), id});
            ASSERT(ok, "unreachable");
            OP_REPO.map.push_back(Op{it_->first, it->second});
            OP_REPO.knownList.erase(it);
            return {id};
        } else {
            RUNTIME_ERROR(fmt::format("Unknown operator \"{}\"", name));
        }
    }
    std::optional<OpType> OpType::tryParse(const char *name) {
        if (auto it = OP_REPO.nameMap.find(name); it != OP_REPO.nameMap.end()) {
            return {{it->second}};
        } else {
            return std::nullopt;
        }
    }

    std::string_view OpType::name() const {
        return OP_REPO.map.at(id).name;
    }
    bool OpType::is(std::string_view rhs) const {
        return OP_REPO.map.at(id).name == rhs;
    }

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

    InferResult Operator::infer(TensorRefs inputs, InferOptions const &options) const {
        return OP_REPO.map.at(opType.id).properties.inference(*this, inputs, options);
    }
    LowerOperator Operator::lower(TensorRefs inputs) const {
        return OP_REPO.map.at(opType.id).properties.lower(*this, inputs);
    }

}// namespace refactor::frontend
