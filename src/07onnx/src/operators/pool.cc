#include "computation/operators/pool.h"
#include "common.h"
#include "pool.hh"

namespace refactor::onnx {
    using Op = Pool;
    using Ty = PoolType;

    Op::Pool(Ty type_,
             Ints kernelShape_,
             OptionalInts dilations_,
             OptionalInts pads_,
             OptionalInts strides_)
        : Operator(),
          type(type_),
          kernelShape(std::move(kernelShape_)),
          dilations(std::move(dilations_)),
          pads(std::move(pads_)),
          strides(std::move(strides_)) {}

    auto Op::build(ModelContext const &, std::string_view opType, Attributes attributes) -> OpBox {
        auto kernelShape = std::move(attributes.at("kernel_shape").ints());
        OptionalInts
            dilations = std::nullopt,
            pads = std::nullopt,
            strides = std::nullopt;
        if (auto it = attributes.find("dilations"); it != attributes.end()) {
            dilations.emplace(std::move(it->second.ints()));
        }
        if (auto it = attributes.find("pads"); it != attributes.end()) {
            pads.emplace(std::move(it->second.ints()));
        }
        if (auto it = attributes.find("strides"); it != attributes.end()) {
            strides.emplace(std::move(it->second.ints()));
        }

        Ty ty;
        if (opType == "onnx::AveragePool") {
            ty = Ty::Average;
        } else if (opType == "onnx::LpPool") {
            ty = Ty::Lp;
        } else if (opType == "onnx::MaxPool") {
            ty = Ty::Max;
            return OpBox(std::make_unique<Op>(Ty::Max, kernelShape, dilations, pads, strides));
        } else {
            UNREACHABLEX(void, "Unsupported global pool operator: {}", opType);
        }

        return OpBox(std::make_unique<Op>(
            ty,
            std::move(kernelShape),
            std::move(dilations),
            std::move(pads),
            std::move(strides)));
    }

    auto Op::typeId(Ty type) -> size_t {
        switch (type) {
            case Ty::Average: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Lp: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Max: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto Op::opTypeId() const -> size_t { return typeId(type); }

    auto Op::opTypeName() const -> std::string_view {
        switch (type) {
            case Ty::Average:
                return "onnx::AveragePool";
            case Ty::Lp:
                return "onnx::LpPool";
            case Ty::Max:
                return "onnx::MaxPool";
            default:
                UNREACHABLE();
        }
    }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(1)

        auto const &input = inputs[0];
        if (!input.dataType.isIeee754()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }

        if (input.rank() != kernelShape.size() + 2) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        SmallInts<4> input_(input.rank() - 2);
        for (auto i : range(2l, input.rank())) {
            EXPECT_VAL(input.shape[i], d)
            input_[i - 2] = d;
        }

        auto res = pool(input_, kernelShape, dilations, pads, strides);
        if (res.isErr()) {
            return Err(InferError(ERROR_MSG(res.unwrapErr())));
        }
        auto output_ = std::move(res.unwrap());
        Shape output(input.rank(), DimExpr(0));
        output[0] = input.shape[0];
        output[1] = input.shape[1];
        std::copy(output_.begin(), output_.end(), output.begin() + 2);
        return Ok(Tensors{Tensor::share(input.dataType, std::move(output), extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::Pool;
        using Ty_ = computation::PoolType;

        Ty_ type_;
        switch (type) {
            case Ty::Average:
                type_ = Ty_::Average;
                break;
            case Ty::Lp:
                type_ = Ty_::Lp;
                break;
            case Ty::Max:
                type_ = Ty_::Max;
                break;
            default:
                UNREACHABLE();
        }

        auto rank = inputs[0].rank();
        decltype(Op_::kernelShape) kernelShape_(kernelShape.size());
        std::transform(kernelShape.begin(), kernelShape.end(),
                       kernelShape_.begin(),
                       [](auto d) { return static_cast<ddim_t>(d); });
        return std::make_unique<Op_>(
            type_,
            ceilMode,
            std::move(kernelShape_),
            computation::PoolAttributes(
                rank - 2,
                dilations ? dilations->data() : nullptr,
                pads ? pads->data() : nullptr,
                strides ? strides->data() : nullptr));
    }

}// namespace refactor::onnx
