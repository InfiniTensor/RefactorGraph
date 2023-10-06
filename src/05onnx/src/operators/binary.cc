#include "binary.hh"
#include "common.h"
#include "common/error_handler.h"
#include "common/range.h"
#include "computation/operators/simple_binary.h"

namespace refactor::onnx {
    using namespace common;

    Binary::Binary(BinaryType type_) : type(type_) {}

    auto Binary::build(std::string_view opType, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "Simple binary operator should not have attributes");

        if (opType == "onnx::Add") {
            return std::make_unique<Binary>(BinaryType::Add);
        }
        if (opType == "onnx::Sub") {
            return std::make_unique<Binary>(BinaryType::Sub);
        }
        if (opType == "onnx::Mul") {
            return std::make_unique<Binary>(BinaryType::Mul);
        }
        if (opType == "onnx::Div") {
            return std::make_unique<Binary>(BinaryType::Div);
        }
        UNREACHABLEX(void, "Unsupported binary operator: {}", opType);
    }

    auto Binary::typeId(BinaryType type) -> size_t {
        switch (type) {
            case BinaryType::Add: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case BinaryType::Sub: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case BinaryType::Mul: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case BinaryType::Div: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto Binary::opTypeId() const -> size_t { return typeId(type); }
    auto Binary::opTypeName() const -> std::string_view {
        switch (type) {
            case BinaryType::Add:
                return "onnx::Add";
            case BinaryType::Sub:
                return "onnx::Sub";
            case BinaryType::Mul:
                return "onnx::Mul";
            case BinaryType::Div:
                return "onnx::Div";
            default:
                UNREACHABLE();
        }
    }

    template<decltype(DataType::internal) T>
    void calculate(BinaryType ty, void *dst, void const *a, void const *b) {
        using T_ = typename primitive_t<T>::type;
        auto a_ = *reinterpret_cast<T_ const *>(a);
        auto b_ = *reinterpret_cast<T_ const *>(b);
        auto dst_ = reinterpret_cast<T_ *>(dst);
        switch (ty) {
            case BinaryType::Add:
                *dst_ = a_ + b_;
                break;
            case BinaryType::Sub:
                *dst_ = a_ - b_;
                break;
            case BinaryType::Mul:
                *dst_ = a_ * b_;
                break;
            case BinaryType::Div:
                *dst_ = a_ / b_;
                break;
            default:
                UNREACHABLE();
        }
    }

    auto Binary::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &a = inputs[0];
        auto const &b = inputs[1];
        auto dataType = a.dataType;
        if (!dataType.isNumberic() || b.dataType != dataType) {
            return Err(InferError(ERROR_MSG("Data type not support")));
        }

        MULTIDIR_BROADCAST((ShapeRefs{a.shape, b.shape}))
        auto ans = Tensor::share(dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }

        auto eleSize = dataType.size();
        auto dst = reinterpret_cast<uint8_t *>(ans->malloc());
        for (auto i : range0_(ans->elementsSize())) {
            auto indices = locateN(ans->shape, i);
            auto a_ = locate1(a, indices),
                 b_ = locate1(b, indices);
            auto dst_ = dst + i * eleSize;
            //-------------------------------------
#define CASE(T)                                     \
    case DataType::T:                               \
        calculate<DataType::T>(type, dst_, a_, b_); \
        break
            //-------------------------------------
            switch (dataType.internal) {
                CASE(F32);
                CASE(F64);
                CASE(I32);
                CASE(I64);
                CASE(I8);
                CASE(I16);
                CASE(U8);
                CASE(U16);
                CASE(U32);
                CASE(U64);
                default:
                    ans->free();
                    break;
            }
        }
        return Ok(Tensors{std::move(ans)});
    }

    auto Binary::lower(TensorRefs) const -> LowerOperator {
        computation::SimpleBinaryType type_;
        switch (type) {
            case BinaryType::Add:
                type_ = computation::SimpleBinaryType::Add;
                break;
            case BinaryType::Sub:
                type_ = computation::SimpleBinaryType::Sub;
                break;
            case BinaryType::Mul:
                type_ = computation::SimpleBinaryType::Mul;
                break;
            case BinaryType::Div:
                type_ = computation::SimpleBinaryType::Div;
                break;
            default:
                break;
        }
        return {std::make_shared<computation::SimpleBinary>(type_), {0, 1}};
    }

}// namespace refactor::onnx
