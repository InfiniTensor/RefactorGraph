#include "moe.hh"
#include "common.h"
#include "computation/operators/moe.h"

namespace refactor::moe {

    AssignPos::AssignPos(uint32_t topk, uint32_t numExperts) : Operator() ,topk(topk), numExperts(numExperts){}

    auto AssignPos::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox { 
        auto topk = attributes["topk"].int_();
        auto num_experts = attributes["num_experts"].int_();
        return OpBox(std::make_unique<AssignPos>(topk, num_experts));
    }
    auto AssignPos::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto AssignPos::opTypeId() const -> size_t { return typeId(); }
    auto AssignPos::opTypeName() const -> std::string_view { return "moe::AssignPos"; }

    auto AssignPos::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(1)

        auto const &gate = inputs[0];
        
        if (gate.dataType != DataType::I16) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        
        return Ok(Tensors{Tensor::share(gate.dataType, Shape{DimExpr(numExperts)}, extractDependency(inputs)),
            Tensor::share(gate.dataType, gate.shape, extractDependency(inputs))});
    }

    auto AssignPos::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::AssignPos;
        return std::make_unique<Op_>(topk, numExperts);
    }

    Reorder::Reorder(bool scatter, uint32_t topk, uint32_t dim) : Operator() ,scatter(scatter), top(topk), dim(dim){}

    auto Reorder::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox { 
        auto topk = attributes["topk"].int_();
        bool scatter = attributes["scatter"].int_() != 0 ;
        bool dim = attributes["dim"].int_();
        return OpBox(std::make_unique<Reorder>(scatter, topk, dim));
    }
    auto Reorder::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Reorder::opTypeId() const -> size_t { return typeId(); }
    auto Reorder::opTypeName() const -> std::string_view { return "moe::Reorder"; }

    auto Reorder::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(2)
        auto const &input = inputs[0];
        auto const &pos = inputs[1];
        if (dim != 0)
            return Err(InferError(ERROR_MSG("dim is not right!")));
        if(scatter && input.elementsSize() * top != pos.elementsSize())
            return Err(InferError(ERROR_MSG("Inputs data size are not right!")));
        else if(!scatter && input.elementsSize()  != pos.elementsSize())
            return Err(InferError(ERROR_MSG("Inputs data size are not right!")));

        if (pos.dataType != DataType::I16) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        
        return Ok(Tensors{Tensor::share(input.dataType, pos.shape, extractDependency(inputs))});
    }

    auto Reorder::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Reorder;
        return std::make_unique<Op_>(scatter, top);
    }
}// namespace refactor::llm
