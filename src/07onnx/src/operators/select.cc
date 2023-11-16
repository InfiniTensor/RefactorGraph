#include "computation/operators/select.h"
#include "common.h"
#include "select.hh"
#include <execution>

namespace refactor::onnx {
    using Op = Select;
    using Ty = SelectType;

    Op::Select(Ty type_)
        : Operator(), type(type_) {}

    auto Op::build(std::string_view opType, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "Select operator should not have attributes");

        auto type = opType == "onnx::Max"   ? Ty::Max
                    : opType == "onnx::Min" ? Ty::Min
                                            : UNREACHABLEX(Ty, "{} not support", opType);
        return OpBox(std::make_unique<Op>(type));
    }

    auto Op::typeId(Ty type) -> size_t {
        switch (type) {
            case Ty::Max: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Min: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto Op::opTypeId() const -> size_t { return typeId(type); }

    auto Op::opTypeName() const -> std::string_view {
        switch (type) {
            case Ty::Max:
                return "onnx::Max";
            case Ty::Min:
                return "onnx::Min";
            default:
                UNREACHABLE();
        }
    }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        ShapeRefs shapes;
        shapes.reserve(inputs.size());
        auto dataType = inputs[0].dataType;
        for (auto &input : inputs) {
            if (input.dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            shapes.emplace_back(input.shape);
        }
        MULTIDIR_BROADCAST(shapes)
        auto ans = Tensor::share(dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }

        auto eleSize = dataType.size();
        auto dst = reinterpret_cast<uint8_t *>(ans->malloc());

        //-------------------------------------
#define CASE(T)                                                                                                              \
    case DataType::T: {                                                                                                      \
        using t = primitive<DataType::T>::type;                                                                              \
        std::vector<t> src(inputs.size());                                                                                   \
        for (auto i : range0_(ans->elementsSize())) {                                                                        \
            auto indices = locateN(ans->shape, i);                                                                           \
            auto dst_ = reinterpret_cast<t *>(dst + i * eleSize);                                                            \
            std::transform(inputs.begin(), inputs.end(), src.begin(),                                                        \
                           [&indices](auto const &input) { return *reinterpret_cast<t const *>(locate1(input, indices)); }); \
            switch (type) {                                                                                                  \
                case Ty::Max:                                                                                                \
                    *dst_ = *std::max_element(src.begin(), src.end());                                                       \
                    break;                                                                                                   \
                case Ty::Min:                                                                                                \
                    *dst_ = *std::min_element(src.begin(), src.end());                                                       \
                    break;                                                                                                   \
                default:                                                                                                     \
                    UNREACHABLE();                                                                                           \
            }                                                                                                                \
        }                                                                                                                    \
    } break;
        //-------------------------------------

        switch (inputs[0].dataType.internal) {
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

        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Select;
        using Ty_ = computation::SelectType;

        Ty_ type_;
        switch (type) {
            case Ty::Max:
                type_ = Ty_::Max;
                break;
            case Ty::Min:
                type_ = Ty_::Min;
                break;
            default:
                UNREACHABLE();
        }

        return std::make_unique<Op_>(type_);
    }

}// namespace refactor::onnx
