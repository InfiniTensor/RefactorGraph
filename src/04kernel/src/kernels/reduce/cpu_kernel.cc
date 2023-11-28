#include "cpu_kernel.hh"
#include "kernel/attributes/transpose_info.h"
#include <numeric>
#include <unordered_set>

namespace refactor::kernel {
    using K = ReduceCpu;
    using DT = DataType;

    K::ReduceCpu(
        decltype(dataType) dataType_,
        decltype(reduceType) reduceType_,
        decltype(axes) axes_,
        decltype(shape) shape_) noexcept
        : Kernel(),
          dataType(dataType_),
          reduceType(reduceType_),
          axes(std::move(axes_)),
          shape(std::move(shape_)) {}

    auto K::build(decltype(axes) axes_, ReduceType reduceType_, TensorRefs inputs_) noexcept -> KernelBox {
        auto const &x = inputs_[0].get();
        return x.dataType.isCpuNumberic()
                   ? std::make_unique<K>(x.dataType, reduceType_, std::move(axes_), x.shape)
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
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
    Routine lowerTyped(Shape shape, Axes axes, ReduceType reduceType) {
        using namespace runtime;
        using dt = typename primitive<T>::type;
        Shape perm;
        std::unordered_set axesSet(axes.begin(), axes.end());
        size_t outsideSize = 1;
        size_t onAxesSize = 1;
        for (auto i : range0_(shape.size())) {
            if (!axesSet.contains(i)) {
                perm.push_back(i);
                outsideSize *= shape[i];
            }
        }
        for (auto axis : axes) {
            perm.push_back(axis);
            onAxesSize *= shape[axis];
        }
        TransposeInfo info = TransposeInfo(shape, perm);
        dt (*accumulate)(dt const a, dt const b);
        switch (reduceType) {
            case ReduceType::Mean:
            case ReduceType::Sum:
                accumulate = [](dt const a, dt const b) { return static_cast<dt>(a + b); };
                break;
            case ReduceType::Max:
                accumulate = [](dt const a, dt const b) { return std::max(a, b); };
                break;
            case ReduceType::Min:
                accumulate = [](dt const a, dt const b) { return std::min(a, b); };
                break;
            default:
                UNREACHABLE();
        }
        dt (*tailInvoke)(dt const a, size_t onAxesSize);
        switch (reduceType) {
            case ReduceType::Mean:
                tailInvoke = [](dt const a, size_t onAxesSize) { return static_cast<dt>(a / onAxesSize); };
                break;
            case ReduceType::Max:
            case ReduceType::Min:
            case ReduceType::Sum:
                tailInvoke = [](dt const a, size_t onAxesSize) { return a; };
                break;
            default:
                UNREACHABLE();
        }
        return [info, outsideSize, onAxesSize, accumulate, tailInvoke](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto input = reinterpret_cast<dt const *>(inputs[0]);
            auto output = reinterpret_cast<dt *>(outputs[0]);
            for (auto i : range0_(outsideSize)) {
                output[i] = 0;
                for (auto j : range0_(onAxesSize)) {
                    auto k = info.locate(i * onAxesSize + j);
                    output[i] = accumulate(output[i], input[k]);
                }
                output[i] = tailInvoke(output[i], onAxesSize);
            }
        };
    }

    auto K::lower(Resources &res) const noexcept -> RoutineWorkspace {

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
