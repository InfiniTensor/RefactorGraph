#include "computation/operators/cum_sum.h"
#include "common.h"
#include "cum_sum.hh"
#include "common.h"
#include <execution>
#include <numeric>

namespace refactor::onnx {
    using Op = CumSum;

    Op::CumSum(bool exclusive_, bool reverse_)
        : Operator(),
          exclusive(exclusive_),
          reverse(reverse_) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        auto exclusive = defaultOr(attributes, "exclusive", {0}).int_() != 0;
        auto reverse = defaultOr(attributes, "reverse", {0}).int_() != 0;
        return OpBox(std::make_unique<Op>(exclusive, reverse));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::CumSum"; }

    template<decltype(DataType::internal) T>
    void accumulate_(void *dst, void const *src, void *acc) {
        using T_ = typename primitive_t<T>::type;
        *reinterpret_cast<T_ *>(dst) = *reinterpret_cast<T_ const *>(src) + *reinterpret_cast<T_ *>(acc);
    }

    void accumulate(DataType dataType, void *dst, void const *src, size_t stepBytes) {
        auto acc = reinterpret_cast<uint8_t *>(dst) - stepBytes;
        switch (dataType.internal) {
#define CASE(T)                                  \
    case DataType::T:                            \
        accumulate_<DataType::T>(dst, src, acc); \
        break;
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
                TODO("DataType not support");
        }
    }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &x = inputs[0];
        auto const &axis = inputs[1];
        if (!axis.shape.empty()) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        auto dataType = x.dataType;
        if (!dataType.isNumberic() ||
            (axis.dataType != DataType::I64 &&
             axis.dataType != DataType::I32)) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto ans = Tensor::share(dataType, x.shape, extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }
        if (reverse) {// TODO: support reverse
            return Ok(Tensors{std::move(ans)});
        }
        auto rank = ans->rank();
        auto axis_ = axis.dataType == DataType::I64
                         ? *axis.data->get<int64_t>()
                         : *axis.data->get<int32_t>();
        if (axis_ < 0) {
            axis_ += rank;
        }
        if (axis_ < 0 || rank <= axis_) {
            return Err(InferError(ERROR_MSG("Invalid axis")));
        }
        auto eleSize = dataType.size();
        if (!reverse) {
            std::for_each_n(std::execution::seq, natural_t(0), ans->elementsSize(),
                            [&, this, axis_, eleSize,
                             step = std::accumulate(ans->shape.begin() + axis_ + 1, ans->shape.end(), eleSize,
                                                    [](auto const acc, auto const &d) { return acc * d.value(); }),
                             src = x.data->get<uint8_t>(),
                             dst = reinterpret_cast<uint8_t *>(ans->malloc())](auto i) {
                                auto indices = locateN(ans->shape, i);
                                auto axisIdx = indices[axis_];
                                auto dst_ = dst + i * eleSize;
                                if (axisIdx == 0) {
                                    if (exclusive) {
                                        std::memset(dst_, 0, eleSize);
                                    } else {
                                        std::memcpy(dst_, src + i * eleSize, eleSize);
                                    }
                                } else {
                                    auto src_ = src + i * eleSize;
                                    if (exclusive) {
                                        accumulate(dataType, dst_, src_ - step, step);
                                    } else {
                                        accumulate(dataType, dst_, src_, step);
                                    }
                                }
                            });
        } else {
            UNREACHABLE();
        }
        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::CumSum;
        return std::make_unique<Op_>(exclusive, reverse);
    }

}// namespace refactor::onnx
