#include "tile.hh"
#include "common.h"
#include "common/range.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;
    using Op = Tile;

    Op::Tile() : Operator() {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "Tile operator should not have attributes");
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Tile"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &input = inputs[0];
        auto const &repeats = inputs[1];

        auto rank = input.rank();
        if (repeats.dataType != DataType::I64 || repeats.shape[0].value() != rank || !repeats.hasData()) {
            return Err(InferError(ERROR_MSG("repeats not support")));
        }
        EXPECT_VAL(repeats.shape[0], repeatsSize)
        ASSERT(repeatsSize == rank, ERROR_MSG("repeats size error"));

        auto repeats_ = repeats.data->get<int64_t>();
        Shape output(rank, DimExpr(1));
        for (auto i : range0_(rank)) {
            if (repeats_[i] == 1) {
                output[i] = input.shape[i];
            } else {
                EXPECT_VAL(input.shape[i], dim)
                output[i] = DimExpr(dim * repeats_[i]);
            }
        }

        auto ans = Tensor::share(input.dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }

        std::for_each_n(std::execution::unseq,
                        natural_t(0), ans->elementsSize(),
                        [&ans, &input, rank,
                         dst = reinterpret_cast<uint8_t *>(ans->malloc()),
                         eleSize = input.dataType.size()](auto i) {
                            auto indices = locateN(ans->shape, i);
                            for (auto j : range0_(rank)) {
                                indices[j] %= input.shape[j].value();
                            }
                            std::memcpy(dst + i * eleSize, locate1(input, indices), eleSize);
                        });
        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs inputs) const -> LowerOperator {
        TODO("");
    }

}// namespace refactor::onnx
