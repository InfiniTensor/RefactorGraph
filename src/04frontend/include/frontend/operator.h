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

        bool isInt() const;
        bool isInts() const;
        bool isFloat() const;
        bool isFloats() const;
        bool isString() const;
        bool isStrings() const;
        bool isTensor() const;
        bool isTensors() const;

        Int &int_();
        Ints &ints();
        Float &float_();
        Floats &floats();
        String &string();
        Strings &strings();
        Tensor_ &tensor();
        Tensors &tensors();

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
    class OpBox;

    class Operator {
    public:
        using InputVec = absl::InlinedVector<size_t, 1>;

        virtual size_t opTypeId() const = 0;
        virtual std::string_view opTypeName() const = 0;
        virtual InputVec valueDependentInputs() const { return {}; }
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

    class OpBox {
        std::unique_ptr<Operator> op;

    public:
        explicit OpBox(std::unique_ptr<Operator> ptr) : op(std::move(ptr)) {}

        Operator *operator->() { return op.get(); }
        Operator const *operator->() const { return op.get(); }
    };

    struct Node {
        OpBox op;
        std::string name;

        template<class T> T const *opAs() const {
            return dynamic_cast<T *>(op.operator->());
        }
    };

}// namespace refactor::frontend

#endif// FRONTEND_OPERATOR_H
