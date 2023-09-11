#include "computation/operator.h"
#include "common/error_handler.h"
#include "computation/graph.h"

namespace refactor::computation {

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

    bool OpType::operator==(OpType const &rhs) const { return id == rhs.id; }
    bool OpType::operator!=(OpType const &rhs) const { return !operator==(rhs); }

    struct Op {
        const char *name;
        InferFn inference;
    };

    struct OpRepo {
        std::vector<Op> map;
        std::unordered_map<const char *, size_t> nameMap;
        std::unordered_map<const char *, InferFn> knownList;
    } static OP_REPO;

    void OpType::register_(const char *name, InferFn infer) {
        if (OP_REPO.nameMap.find(name) != OP_REPO.nameMap.end() ||
            OP_REPO.knownList.find(name) != OP_REPO.knownList.end()) {
            RUNTIME_ERROR("Operator already registered");
        }
        OP_REPO.knownList[name] = infer;
    }
    OpType OpType::parse(const char *name) {
        if (auto it = OP_REPO.nameMap.find(name); it != OP_REPO.nameMap.end()) {
            return {it->second};
        } else if (auto it = OP_REPO.knownList.find(name); it != OP_REPO.knownList.end()) {
            size_t id = OP_REPO.map.size();
            OP_REPO.map.push_back(Op{name, it->second});
            OP_REPO.nameMap[name] = id;
            OP_REPO.knownList.erase(it);
            return {id};
        } else {
            RUNTIME_ERROR("Unknown operator");
        }
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

    InferResult Operator::infer(Edges inputs) const {
        return OP_REPO.map.at(opType.id).inference(*this, inputs);
    }

}// namespace refactor::computation
