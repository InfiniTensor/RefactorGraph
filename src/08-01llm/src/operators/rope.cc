#include "rope.hh"
#include "common.h"
#include "computation/operators/rope.h"
namespace refactor::llm {
    using Op = RotaryPositionEmbedding;

    Op::RotaryPositionEmbedding(float _theta) : Operator(), theta(_theta) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto _theta = attributes.getOrInsert("theta", {10000}).float_();
        return OpBox(std::make_unique<Op>(_theta));
    }

    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "llm::RotaryPositionEmbedding"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(2)

        auto input = inputs[0];  // (batchsize?, seq_len, n_heads, head_dim)
        auto pos_ids = inputs[1];// (batchsize?, seq_len)

        // Check shapes
        if (input.rank() == 4) {
            if (pos_ids.rank() != 2) {
                return Err(InferError(ERROR_MSG("Invalid pos_ids shape")));
            }
            if (input.shape[0] != pos_ids.shape[0]) {
                return Err(InferError(ERROR_MSG("Batchsizes not matched")));
            }
            if (input.shape[1] != pos_ids.shape[1]) {
                return Err(InferError(ERROR_MSG("Seq_len not matched")));
            }
        } else if (input.rank() == 3) {
            if (pos_ids.rank() != 1) {
                return Err(InferError(ERROR_MSG("Invalid pos_ids shape")));
            }
            if (input.shape[0] != pos_ids.shape[0]) {
                return Err(InferError(ERROR_MSG("Seq_len not matched")));
            }
        } else {
            return Err(InferError(ERROR_MSG("Invalid input shape")));
        }

        // Check types
        if (input.dataType != DataType::F32 && input.dataType != DataType::FP16) {
            return Err(InferError(ERROR_MSG("Invalid input dtype")));
        }
        if (pos_ids.dataType != DataType::I64) {
            return Err(InferError(ERROR_MSG("Invalid pos_ids dtype")));
        }

        return Ok(Tensors{Tensor::share(input.dataType, input.shape, extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::RotaryPositionEmbedding;
        return std::make_unique<Op_>(theta);
    }

}// namespace refactor::llm
