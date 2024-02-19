#ifndef FRONTEND_OPERATOR_H
#define FRONTEND_OPERATOR_H

#include "computation/operator.h"
#include "infer.h"
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
    class Attributes {
        std::unordered_map<std::string, Attribute> map;

    public:
        void insert(std::string, Attribute);
        bool empty() const;
        Attribute &operator[](const char *);
        Attribute const &operator[](const char *) const;
        std::optional<std::reference_wrapper<Attribute>> get(const char *);
        std::optional<std::reference_wrapper<Attribute const>> get(const char *) const;
        Attribute &getOrInsert(const char *, Attribute);
    };
    using ModelContext = std::unordered_map<std::string, Attribute>;

    class Operator;
    class OpBox;

    class Operator {
    public:
        using InputVec = absl::InlinedVector<size_t, 1>;

        virtual ~Operator() = default;
        virtual size_t opTypeId() const = 0;
        virtual std::string_view opTypeName() const = 0;
        virtual InputVec valueDependentInputs() const { return {}; }
        virtual InferResult infer(TensorRefs, InferOptions const &) const = 0;
        virtual computation::OpBox lower(TensorRefs) const;

        template<class T, class... Args>
        bool is(Args &&...args) const noexcept {
            return this->opTypeId() == T::typeId(std::forward<Args>(args)...);
        }

        using Builder = OpBox (*)(ModelContext const &, std::string_view, Attributes);
        static void register_(std::string, Builder);
        template<class T> static void register_(std::string opTypeName) {
            register_(std::move(opTypeName), T::build);
        }

        static OpBox build(ModelContext const &, std::string, Attributes);
    };

    class OpBox {
        std::unique_ptr<Operator> _op;

    public:
        explicit OpBox(decltype(_op));

        Operator *operator->();
        Operator const *operator->() const;
    };

    struct Node {
        OpBox op;
        std::string name;

        template<class T> T const *opAs() const {
            return dynamic_cast<T *>(op.operator->());
        }
    };

    using ShapeResult = Result<Shape, std::string>;
    using ShapeRefs = std::vector<std::reference_wrapper<Shape const>>;

    /// @brief 多方向形状广播。
    /// @param inputs 所有输入的形状。
    /// @return 广播后的形状。
    ShapeResult multidirBroadcast(ShapeRefs const &);

    /// @brief 单方向形状广播。
    /// @param target 目标形状。
    /// @param test 测试形状。
    /// @return 测试形状是否可以广播到目标形状。
    bool unidirBroadcast(Shape const &target, Shape const &test);

#define EXPECT_NO_ATTRI \
    ASSERT(attributes.empty(), "{} operator should not have attributes", opType)

#define EXPECT_SIZE(N)                                         \
    if (inputs.size() != (N)) {                                \
        return Err(InferError(ERROR_MSG("Input size error"))); \
    }

#define EXPECT_VAL(DIM, VAL)                                             \
    int64_t VAL;                                                         \
    if ((DIM).hasValue()) {                                              \
        VAL = (DIM).value();                                             \
    } else {                                                             \
        return Err(InferError(UnknownVariable{(DIM.variable()->name)})); \
    }

#define MULTIDIR_BROADCAST(SHAPES)                              \
    Shape output;                                               \
    {                                                           \
        auto res = multidirBroadcast(SHAPES);                   \
        if (res.isErr()) {                                      \
            return Err(InferError(ERROR_MSG(res.unwrapErr()))); \
        }                                                       \
        output = std::move(res.unwrap());                       \
    }

}// namespace refactor::frontend

#endif// FRONTEND_OPERATOR_H
