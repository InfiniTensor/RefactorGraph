#include "simple_binary.hh"
#include "common.h"
#include "computation/operators/simple_binary.h"
#include "kernel/collectors/simple_binary.h"
#include "runtime/resource.h"
#include <execution>

namespace refactor::onnx {
    using Op = SimpleBinary;
    using Ty = SimpleBinaryType;

    Op::SimpleBinary(Ty type_)
        : Operator(), type(type_) {}

    auto Op::build(ModelContext const &, std::string_view opType, Attributes attributes) -> OpBox {
        auto fmod = attributes.getOrInsert("fmod", {0}).int_();
        // clang-format off
        auto type =
            opType == "onnx::Add" ? Ty::Add :
            opType == "onnx::Sub" ? Ty::Sub :
            opType == "onnx::Mul" ? Ty::Mul :
            opType == "onnx::Div" ? Ty::Div :
            opType == "onnx::Pow" ? Ty::Pow :
            opType == "onnx::And" ? Ty::And :
            opType == "onnx::Or"  ? Ty::Or  :
            opType == "onnx::Xor" ? Ty::Xor :
            opType == "onnx::Mod" ? (fmod == 0 ? Ty::Mod : Ty::Fmod) :
            UNREACHABLEX(Ty, "Unsupported binary operator: {}", opType);
        // clang-format on
        return OpBox(std::make_unique<Op>(type));
    }

    auto Op::typeId(Ty type) -> size_t {
        switch (type) {
            case Ty::Add: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Sub: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Mul: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Div: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Pow: {
                static uint8_t ID = 5;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::And: {
                static uint8_t ID = 6;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Or: {
                static uint8_t ID = 7;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Xor: {
                static uint8_t ID = 8;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Mod: {
                static uint8_t ID = 9;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Fmod: {
                static uint8_t ID = 10;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto Op::opTypeId() const -> size_t { return typeId(type); }
    auto Op::opTypeName() const -> std::string_view {
        // clang-format off
        switch (type) {
            case Ty::Add: return "onnx::Add";
            case Ty::Sub: return "onnx::Sub";
            case Ty::Mul: return "onnx::Mul";
            case Ty::Div: return "onnx::Div";
            case Ty::Pow: return "onnx::Pow";
            case Ty::And: return "onnx::And";
            case Ty::Or : return "onnx::Or" ;
            case Ty::Xor: return "onnx::Xor";
            case Ty::Mod: return "onnx::Mod";
            case Ty::Fmod: return "onnx::Mod";
            default: UNREACHABLE();
        }
        // clang-format on
    }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &a = inputs[0];
        auto const &b = inputs[1];
        auto dataType = a.dataType;
        if (type == Ty::Pow) {
            if (!dataType.isFloat() || !b.dataType.isNumberic()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else {
            if (!dataType.isNumberic() || b.dataType != dataType) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        }

        MULTIDIR_BROADCAST((ShapeRefs{a.shape, b.shape}))
        auto ans = Tensor::share(dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans}) || type == Ty::Pow) {
            return Ok(Tensors{std::move(ans)});
        }

        {
            using Shape = kernel::Shape;
            using Tensor = kernel::Tensor;
            using LayoutType = kernel::LayoutType;

            Shape t1Shape(a.shape.size(), 1);
            Shape t2Shape(b.shape.size(), 1);
            Shape oShape(ans->shape.size(), 1);
            std::transform(std::execution::unseq,
                           a.shape.begin(), a.shape.end(), t1Shape.begin(),
                           [](auto const &i) { return static_cast<dim_t>(i.value()); });
            std::transform(std::execution::unseq,
                           b.shape.begin(), b.shape.end(), t2Shape.begin(),
                           [](auto const &i) { return static_cast<dim_t>(i.value()); });
            auto t1 = Tensor::share(a.dataType, t1Shape, LayoutType::Others, a.data);
            auto t2 = Tensor::share(b.dataType, t2Shape, LayoutType::Others, b.data);
            std::transform(std::execution::unseq,
                           ans->shape.begin(), ans->shape.end(), oShape.begin(),
                           [](auto const &i) { return static_cast<dim_t>(i.value()); });
            auto o = Tensor::share(a.dataType, oShape, LayoutType::Others);
            runtime::Resources res;
            auto type_ = static_cast<kernel::SimpleBinaryType>(type);
            const auto collector = kernel::SimpleBinaryCollector(computation::Target::Cpu, type_);
            auto routine = std::move(collector.filter({*t1, *t2}, {*o}).at(0))->lower(res).routine;
            void const *inputsCpu[]{*t1->data, *t2->data};
            void *outputsCpu[]{o->malloc()};
            routine(res, nullptr, inputsCpu, outputsCpu);
            ans->data = o->data;
        }

        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::SimpleBinary;
        using Ty_ = computation::SimpleBinaryType;
        Ty_ type_;
        // clang-format off
        switch (type) {
            case Ty::Add : type_ = Ty_::Add; break;
            case Ty::Sub : type_ = Ty_::Sub; break;
            case Ty::Mul : type_ = Ty_::Mul; break;
            case Ty::Div : type_ = Ty_::Div; break;
            case Ty::Pow : type_ = Ty_::Pow; break;
            case Ty::And : type_ = Ty_::And; break;
            case Ty::Or  : type_ = Ty_::Or ; break;
            case Ty::Xor : type_ = Ty_::Xor; break;
            case Ty::Mod : type_ = Ty_::Mod; break;
            case Ty::Fmod : type_ = Ty_::Fmod; break;
            default: UNREACHABLE();
        }
        // clang-format on
        return std::make_unique<Op_>(type_);
    }

}// namespace refactor::onnx
