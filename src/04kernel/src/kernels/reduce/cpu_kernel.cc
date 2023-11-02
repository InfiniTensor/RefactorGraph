#include "cpu_kernel.hh"
#include "kernel/attributes/transpose_info.h"
#include "runtime/mem_manager.hh"
#include <numeric>
#include <unordered_set>

namespace refactor::kernel {
    using K = ReduceCpu;
    using DT = DataType;

    K::ReduceCpu(decltype(axes) axes_, ReduceType reduceType_, DataType dataType_, Shape shape_) noexcept
        : Kernel(),
          axes(axes_),
          reduceType(reduceType_),
          dataType(dataType_),
          shape(shape_) {}

    auto K::build(decltype(axes) axes_, ReduceType reduceType_, TensorRefs inputs_) noexcept -> KernelBox {
        auto const &x = inputs_[0].get();
        return x.dataType.isCpuNumberic()
                   ? std::make_unique<K>(axes_, reduceType_, x.dataType, x.shape)
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
        // 这里 ID 是什么含义？
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing reduce operation on generic cpu";
    }

    template<decltype(DT::internal) T>
    Routine lowerTyped(Shape shape, std::vector<int64_t> axes, ReduceType reduceType) {
        using namespace runtime;
        using dt = typename primitive<T>::type;
        Shape perm;
        std::unordered_set axesSet(axes.begin(), axes.end());
        size_t outsideSize = 1;
        size_t onAxesSize = 1;
        for (auto i : range0_(shape.size())) {
            if (axesSet.find(i) == axesSet.end()) {
                perm.push_back(i);
                outsideSize *= shape[i];
            }
        }
        for (auto axis : axes) {
            perm.push_back(axis);
            onAxesSize *= shape[axis];
        }
        TransposeInfo info = TransposeInfo(shape, perm);
        auto accumulate = [reduceType](dt const a, dt const b) {
            switch (reduceType) {
                case ReduceType::Mean:
                case ReduceType::Sum:
                    return static_cast<dt>(a + b);
                case ReduceType::Max:
                    return std::max(a, b);
                case ReduceType::Min:
                    return std::min(a, b);
                default:
                    UNREACHABLE();
            }
        };
        auto tailInvoke = [reduceType, onAxesSize](dt const a) {
            switch (reduceType) {
                case ReduceType::Mean:
                    return static_cast<dt>(a / onAxesSize);
                    break;
                case ReduceType::Max:
                case ReduceType::Min:
                case ReduceType::Sum:
                    return a;
                    break;
                default:
                    UNREACHABLE();
            }
        };
        return [info, outsideSize, onAxesSize, accumulate, tailInvoke](Resources &res, void const **inputs, void **outputs) {
            auto input = reinterpret_cast<dt const *>(inputs[0]);
            auto output = reinterpret_cast<dt *>(outputs[0]);
            for (auto i : range0_(outsideSize)) {
                output[i] = 0;
                for (auto j : range0_(onAxesSize)) {
                    auto k = info.locate(i * onAxesSize + j);
                    output[i] = accumulate(output[i], input[k]);
                }
                output[i] = tailInvoke(output[i]);
            }
        };
    }

    auto K::lower(Resources &res) const noexcept -> Routine {

#define CASE(T) \
    case T:     \
        return lowerTyped<DataType::T>(shape, axes, reduceType)

        switch (dataType) {
            CASE(DataType::U32);
            CASE(DataType::U64);
            CASE(DataType::I32);
            CASE(DataType::I64);
            CASE(DataType::F32);
            CASE(DataType::F64);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
