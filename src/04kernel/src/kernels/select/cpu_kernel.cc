#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = SelectCpu;
    using DT = DataType;

    K::SelectCpu(
        decltype(dataType) dataType_,
        decltype(selectType) selectType_,
        decltype(broadcaster) broadcaster_,
        decltype(inputsNum) inputsNum_) noexcept
        : dataType(dataType_),
          selectType(selectType_),
          broadcaster(broadcaster_),
          inputsNum(inputsNum_) {}

    auto K::build(SelectType selectType_, TensorRefs inputs_) noexcept -> KernelBox {
        auto const &x = inputs_[0].get();
        return x.dataType.isCpuNumberic()
                   ? std::make_unique<K>(x.dataType, selectType_, Broadcaster(inputs_), inputs_.size())
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
        return "Performing select operation on generic cpu";
    }

    template<class T>
    auto lowerTyped(SelectType selectType, Broadcaster broadcaster, size_t inputsNum) noexcept -> RoutineWorkspace {
        using namespace runtime;

        T(*op)
        (T const a, T const b);
        switch (selectType) {
            case SelectType::Max:
                op = [](T const a, T const b) { return std::max(a, b); };
                break;
            case SelectType::Min:
                op = [](T const a, T const b) { return std::min(a, b); };
                break;
            default:
                UNREACHABLE();
        }

        if (broadcaster.needBroadcast()) {
            return [n = broadcaster.outputsCount, inputsNum, op](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
                auto output = reinterpret_cast<T *>(outputs[0]);
                for (auto i : range0_(n)) {
                    for (auto inputIdx : range0_(inputsNum)) {
                        auto input = reinterpret_cast<const T *>(inputs[inputIdx]);
                        if (inputIdx == 0) {
                            output[i] = input[i];
                        } else {
                            output[i] = op(output[i], input[i]);
                        }
                    }
                };
            };
        } else {
            return [broadcaster, inputsNum, op](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
                auto output = reinterpret_cast<T *>(outputs[0]);
                for (auto i : range0_(broadcaster.outputsCount)) {
                    std::vector<dim_t> ans(broadcaster.inputsCount);
                    broadcaster.locate(i, ans.data());
                    for (auto inputIdx : range0_(inputsNum)) {
                        auto input = reinterpret_cast<const T *>(inputs[inputIdx]);
                        if (inputIdx == 0) {
                            output[i] = input[ans[inputIdx]];
                        } else {
                            output[i] = op(output[i], input[ans[inputIdx]]);
                        }
                    }
                }
            };
        }
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
#define CASE(DT)       \
    case DataType::DT: \
        return lowerTyped<primitive<DataType::DT>::type>(selectType, broadcaster, inputsNum)

        switch (dataType) {
            CASE(F32);
            CASE(U8);
            CASE(I8);
            CASE(U16);
            CASE(I16);
            CASE(I32);
            CASE(I64);
            CASE(F64);
            CASE(U32);
            CASE(U64);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
