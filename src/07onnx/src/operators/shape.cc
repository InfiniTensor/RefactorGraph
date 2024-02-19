#include "shape.hh"
#include "common.h"

namespace refactor::onnx {
    using Op = Shape;

    Op::Shape(Int start_, std::optional<Int> end_)
        : Operator(),
          start(start_),
          end(end_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto start = attributes.getOrInsert("start", {0}).int_();
        std::optional<Int> end = std::nullopt;
        if (auto opt = attributes.get("end"); opt) {
            end.emplace(opt->get().int_());
        }
        return OpBox(std::make_unique<Op>(start, end));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Shape"; }
    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(1)

        auto const &data = inputs[0];
        auto const rank = data.rank();

        auto start_ = start < 0 ? start + rank : start,
             end_ = end ? (*end < 0 ? *end + rank : *end) : rank;
        if (start_ < 0 || rank <= start_) {
            return Err(InferError(ERROR_MSG("start out of range")));
        }
        if (end_ <= start || rank < end_) {
            return Err(InferError(ERROR_MSG("end out of range")));
        }

        auto ans = Tensor::share(DataType::I64, frontend::Shape{DimExpr(end_ - start_)}, extractDependency(inputs));
        auto dst = reinterpret_cast<int64_t *>(ans->malloc());
        for (auto i : range(start_, end_)) {
            EXPECT_VAL(data.shape[i], dim)
            dst[i - start_] = dim;
        }
        return Ok(Tensors{std::move(ans)});
    }
}// namespace refactor::onnx
