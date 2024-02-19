#include "computation/operators/gather.h"
#include "common.h"
#include "gather.hh"
#include "kernel/collectors/gather.h"
#include "runtime/resource.h"
#include <execution>

namespace refactor::onnx {
    using Op = Gather;

    Op::Gather(Int axis_)
        : Operator(), axis(axis_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto axis = attributes.getOrInsert("axis", {0}).int_();
        return OpBox(std::make_unique<Op>(axis));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Gather"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &data = inputs[0];
        auto const &indices = inputs[1];
        if (indices.dataType != DataType::I32 && indices.dataType != DataType::I64) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }

        auto const rank = data.rank();
        auto axis_ = axis < 0 ? axis + rank : axis;
        if (axis_ < 0 || rank <= axis_) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        auto output = data.shape;
        output.erase(output.begin() + axis_);
        output.insert(output.begin() + axis_, indices.shape.begin(), indices.shape.end());
        auto ans = Tensor::share(data.dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }
        {
            using Shape = kernel::Shape;
            using Tensor = kernel::Tensor;
            using LayoutType = kernel::LayoutType;

            Shape t1Shape(data.shape.size(), 1);
            Shape t2Shape(indices.shape.size(), 1);
            Shape oShape(ans->shape.size(), 1);
            std::transform(std::execution::unseq,
                           data.shape.begin(), data.shape.end(), t1Shape.begin(),
                           [](auto const &i) { return static_cast<dim_t>(i.value()); });
            std::transform(std::execution::unseq,
                           indices.shape.begin(), indices.shape.end(), t2Shape.begin(),
                           [](auto const &i) { return static_cast<dim_t>(i.value()); });
            auto t1 = Tensor::share(data.dataType, t1Shape, LayoutType::Others, data.data);
            auto t2 = Tensor::share(indices.dataType, t2Shape, LayoutType::Others, indices.data);
            std::transform(std::execution::unseq,
                           ans->shape.begin(), ans->shape.end(), oShape.begin(),
                           [](auto const &i) { return static_cast<dim_t>(i.value()); });
            auto o = Tensor::share(data.dataType, oShape, LayoutType::Others);
            runtime::Resources res;
            const auto collector = kernel::GatherCollector(computation::Target::Cpu, axis_);
            auto routine = std::move(collector.filter({*t1, *t2}, {*o}).at(0))->lower(res).routine;
            void const *inputsCpu[]{*t1->data, *t2->data};
            void *outputsCpu[]{o->malloc()};
            routine(res, nullptr, inputsCpu, outputsCpu);
            ans->data = o->data;
        }

        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::Gather;
        auto rank = inputs[0].rank();
        return std::make_unique<Op_>(axis < 0 ? axis + rank : axis, rank);
    }

}// namespace refactor::onnx
