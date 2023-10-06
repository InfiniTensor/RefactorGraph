#ifndef FRONTEND_OPERATOR_H
#define FRONTEND_OPERATOR_H

#include "infer.h"
#include "lower.h"
#include "tensor.h"
#include <variant>

namespace refactor::frontend {

    using Int = int64_t;
    using Ints = std::vector<int64_t>;
    using Float = float;
    using Floats = std::vector<float>;
    using String = std::string;
    using Strings = std::vector<std::string>;
    using Tensors = std::vector<Tensor_>;

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

    class Operator;
    using OpBox = std::unique_ptr<Operator>;
    class Operator {
    public:
        virtual size_t opTypeId() const = 0;
        virtual std::string_view opTypeName() const = 0;
        virtual InferResult infer(TensorRefs, InferOptions const &) const = 0;
        virtual LowerOperator lower(TensorRefs) const { UNREACHABLE(); }

        bool typeEqual(size_t opTypeId) const noexcept {
            return this->opTypeId() == opTypeId;
        }

        using Builder = OpBox (*)(std::string_view, Attributes);
        static OpBox build(std::string, Attributes);
        static void register_(std::string, Builder);
        template<class T> static void register_(std::string opTypeName) {
            register_(std::move(opTypeName), T::build);
        }
    };

    struct Node {
        OpBox op;
        std::string name;

        template<class T> T *opAs() const {
            return dynamic_cast<T *>(op.get());
        }
    };

}// namespace refactor::frontend

#endif// FRONTEND_OPERATOR_H
