#include "common.h"
#include "common/range.h"
#include "computation/operators/simple_binary.h"

namespace refactor::onnx {
    using namespace common;

    enum class Ty {
        Add,
        Sub,
        Mul,
        Div
    };

    template<decltype(DataType::internal) T>
    void calculate(Ty ty, void *dst, void *a, void *b) {
        using T_ = typename primitive_t<T>::type;
        auto a_ = *reinterpret_cast<T_ *>(a);
        auto b_ = *reinterpret_cast<T_ *>(b);
        auto dst_ = reinterpret_cast<T_ *>(dst);
        switch (ty) {
            case Ty::Add:
                *dst_ = a_ + b_;
                break;
            case Ty::Sub:
                *dst_ = a_ - b_;
                break;
            case Ty::Mul:
                *dst_ = a_ * b_;
                break;
            case Ty::Div:
                *dst_ = a_ / b_;
                break;
            default:
                RUNTIME_ERROR("Unreachable");
        }
    }

    InferResult inferArithmetic(Operator const &op, TensorRefs inputs, InferOptions options) {
        EXPECT_SIZE(2)

        auto const &a = inputs[0];
        auto const &b = inputs[1];
        auto dataType = a.dataType;
        if (!dataType.isNumberic() || b.dataType != dataType) {
            return Err(InferError(ERROR_MSG("Data type not support")));
        }

        MULTIDIR_BROADCAST((ShapeRefs{a.shape, b.shape}))
        auto ans = Tensor::share(dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, ans->shape)) {
            return Ok(Tensors{std::move(ans)});
        }

        auto eleSize = dataType.size();
        auto dst = reinterpret_cast<uint8_t *>(ans->malloc());
        for (auto i : range0_(ans->elementsSize())) {
            auto ty = op.opType.is("onnx::Add")   ? Ty::Add
                      : op.opType.is("onnx::Sub") ? Ty::Sub
                      : op.opType.is("onnx::Mul") ? Ty::Mul
                      : op.opType.is("onnx::Div") ? Ty::Div
                                                  : UNREACHABLE();
            auto indices = locateN(ans->shape, i);
            auto a_ = locate1(a, indices),
                 b_ = locate1(b, indices);
            auto dst_ = dst + i * eleSize;
            //-------------------------------------
#define CASE(T)                                   \
    case DataType::T:                             \
        calculate<DataType::T>(ty, dst_, a_, b_); \
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

    static computation::SimpleBinaryType unsupport(OpType opType) {
        RUNTIME_ERROR(fmt::format("{} not support in binary lowering", opType.name()));
    }

    computation::SharedOp lowerArithmetic(Operator const &op, TensorRefs) {
        using namespace computation;

        auto type = op.opType.is("onnx::Add")   ? SimpleBinaryType::Add
                    : op.opType.is("onnx::Sub") ? SimpleBinaryType::Sub
                    : op.opType.is("onnx::Mul") ? SimpleBinaryType::Mul
                    : op.opType.is("onnx::Div") ? SimpleBinaryType::Div
                                                : unsupport(op.opType);
        return std::make_shared<SimpleBinary>(type);
    }
}// namespace refactor::onnx
