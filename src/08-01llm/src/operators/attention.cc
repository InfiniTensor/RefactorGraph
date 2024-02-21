#include "computation/operators/attention.h"
#include "attention.hh"
#include "common.h"

namespace refactor::llm {
    using Op = Attention;

    Op::Attention(decltype(maxSeqLen) maxSeqLen_)
        : Operator(), maxSeqLen(maxSeqLen_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto maxSeqLen = attributes.getOrInsert("max_seq_len", {0}).int_();
        return OpBox(std::make_unique<Op>(maxSeqLen));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "llm::Attention"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        if (inputs.size() < 3) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto const &query = inputs[0],
                   &key = inputs[1],
                   &value = inputs[2];
        auto dt = query.dataType;
        if (key.dataType != dt || value.dataType != dt) {
            return Err(InferError(ERROR_MSG("Data type error")));
        }
        if (query.rank() != 4 || key.rank() != 4 || value.rank() != 4) {
            return Err(InferError(ERROR_MSG("Input Rank error")));
        }
        auto kvShape = Shape{
            query.shape[0],
            key.shape[1],
            query.shape[2],
            query.shape[3],
        };
        if (query.shape[1].hasValue() && kvShape[1].hasValue()) {
            auto nHead = query.shape[1].value(),
                 nkvHead = kvShape[1].value();
            if (nHead < nkvHead || nHead % nkvHead != 0) {
                return Err(InferError(ERROR_MSG("Head size error")));
            }
        }
        if (key.shape != kvShape || value.shape != kvShape) {
            return Err(InferError(ERROR_MSG("KV size error")));
        }
        auto outputs = [&, dt](int64_t cacheSeqLen) {
            auto cacheShape = Shape{
                key.shape[0],
                key.shape[1],
                DimExpr(cacheSeqLen),
                key.shape[3],
            };
            auto deps = extractDependency(inputs);
            return Ok(Tensors{
                Tensor::share(dt, query.shape, deps),
                Tensor::share(dt, cacheShape, deps),
                Tensor::share(dt, cacheShape, deps),
            });
        };
        EXPECT_VAL(query.shape[2], seqlen)

        switch (inputs.size()) {
            case 3:
                if (maxSeqLen <= 0) {
                    return Ok(Tensors{Tensor::share(dt, query.shape, extractDependency(inputs))});
                } else if (maxSeqLen >= seqlen) {
                    return outputs(maxSeqLen);
                } else {
                    return Err(InferError(ERROR_MSG("max_seq_len must not less than seqlen")));
                }
            case 4: {
                auto const &pastSeqLen = inputs[3];
                if (pastSeqLen.dataType != DataType::I64 || pastSeqLen.shape != Shape{DimExpr(1)}) {
                    return Err(InferError(ERROR_MSG("Past seqlen error")));
                }
                if (maxSeqLen <= 0) {
                    auto pastSeqLenVal = pastSeqLen.data->get<int64_t>()[0];
                    return outputs(pastSeqLenVal + seqlen);
                } else if (maxSeqLen >= 1 + seqlen) {
                    return outputs(maxSeqLen);
                } else {
                    return Err(InferError(ERROR_MSG("max_seq_len must not less than seqlen")));
                }
            }
            case 6: {
                auto const &pastSeqLen = inputs[3];
                if (pastSeqLen.dataType != DataType::I64 || pastSeqLen.shape != Shape{DimExpr(1)}) {
                    return Err(InferError(ERROR_MSG("Past seqlen error")));
                }

                auto const &kCahce = inputs[4],
                           &vCache = inputs[5];
                EXPECT_VAL(kCahce.shape[2], kCacheSeqLen)
                EXPECT_VAL(vCache.shape[2], vCacheSeqLen)
                if (kCahce.dataType != dt || vCache.dataType != dt ||
                    kCahce.rank() != 4 || vCache.rank() != 4 ||
                    kCahce.shape[0] != kvShape[0] ||
                    kCahce.shape[2] != kvShape[2] ||
                    kCahce.shape[3] != kvShape[3] ||
                    kCahce.shape[0] != kvShape[0] ||
                    kCahce.shape[2] != kvShape[2] ||
                    kCahce.shape[3] != kvShape[3]) {
                    return Err(InferError(ERROR_MSG("KV cache error")));
                }

                if (maxSeqLen <= 0) {
                    auto pastSeqLenVal = pastSeqLen.data->get<int64_t>()[0];
                    return outputs(pastSeqLenVal + seqlen);
                } else if (maxSeqLen >= 1 + seqlen) {
                    return outputs(maxSeqLen);
                } else {
                    return Err(InferError(ERROR_MSG("max_seq_len must not less than seqlen")));
                }
            }
            default:
                UNREACHABLEX(void, "Attention operator should have 3, 4 or 6 inputs");
        }
        UNREACHABLE();
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Attention;
        return std::make_unique<Op_>();
    }

}// namespace refactor::llm
