#include "expand.hh"
#include "common.h"
#include "computation/operators/broadcast.h"
#include <execution>

namespace refactor::onnx {
    using Op = Expand;

    auto Op::build(ModelContext const &, std::string_view opType, Attributes attributes) -> OpBox {
        EXPECT_NO_ATTRI;
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Expand"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &data = inputs[0];
        auto const &shape = inputs[1];

        if (shape.dataType != DataType::I64 || shape.rank() != 1 || !shape.data) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        }

        auto shape_ = shape.data->get<int64_t>();
        EXPECT_VAL(shape.shape[0], shapeSize)

        Shape forRef(shape_, shape_ + shapeSize);
        MULTIDIR_BROADCAST((ShapeRefs{data.shape, forRef}))
        auto ans = Tensor::share(data.dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }

        std::for_each_n(std::execution::unseq,
                        natural_t(0), ans->elementsSize(),
                        [&data, &ans,
                         dst = reinterpret_cast<uint8_t *>(ans->malloc()),
                         eleSize = data.dataType.size()](auto const i) {
                            std::memcpy(dst + i * eleSize, locate1(data, locateN(ans->shape, i)), eleSize);
                        });
        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Broadcast;
        return std::make_unique<Op_>();
    }

}// namespace refactor::onnx
