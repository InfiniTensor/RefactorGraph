#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = PadCpu;

    K::PadCpu(PadInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(PadInfo info) noexcept -> KernelBox {
        if (info.mode != PadType::Constant) {
            return nullptr;
        }
        return std::make_unique<K>(std::move(info));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing pad operation on generic cpu";
    }

    template<class T>
    static Routine lowerTyped(PadInfo info) {
        using namespace runtime;
        return [info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto x = reinterpret_cast<T const *>(inputs[0]);
            auto const_value = info.have_value ? reinterpret_cast<T const *>(inputs[2])[0] : static_cast<T>(0);
            auto y = reinterpret_cast<T *>(outputs[0]);
            auto getValue = [&](auto tid) {
                int offset = 0;
                for (int i = info.rank - 1; i >= 0; --i) {
                    auto wholePos = tid % info.wholeNDim[i];
                    auto pos = wholePos - info.pads[i];
                    // if pos belongs to pad range, then return -1
                    if (pos < 0 || pos >= info.partNDim[i]) { return -1; }
                    tid = tid / info.wholeNDim[i];
                    offset += pos * info.partStride[i];
                }
                return offset;
            };
            std::for_each_n(std::execution::par_unseq, natural_t(0), info.size, [&](auto i) {
                auto axis = getValue(i);
                y[i] = axis < 0 ? const_value : x[axis];
            });
        };
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
#define CASE_DT(T)    \
    case DataType::T: \
        return lowerTyped<primitive<DataType::T>::type>(std::move(info));
        switch (info.type) {
            CASE_DT(U8)
            CASE_DT(I8)
            CASE_DT(U16)
            CASE_DT(I16)
            CASE_DT(U32)
            CASE_DT(I32)
            CASE_DT(U64)
            CASE_DT(I64)
            CASE_DT(F32)
            CASE_DT(F64)
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
