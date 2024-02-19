#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = PadCpu;

    K::PadCpu(PadInfo info_, PadType mode_, size_t value_) noexcept
        : Kernel(), info(std::move(info_)), mode(mode_), valueLength(value_) {}

    auto K::build(PadInfo info, PadType mode, std::optional<std::reference_wrapper<Tensor const>> value_) noexcept -> KernelBox {
        if (mode != PadType::Constant) {
            return nullptr;
        }
        size_t value = value_ ? value_->get().dataType.size() : 0;
        return std::make_unique<K>(std::move(info), mode, value);
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


    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;

        return [info = this->info, value = this->valueLength](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto src = reinterpret_cast<uint8_t const *>(inputs[0]);
            auto dst = reinterpret_cast<uint8_t *>(outputs[0]);
            std::vector<uint8_t> defaultValue(info.blockSize, 0);
            if (value != 0) {
                auto constValue = reinterpret_cast<uint8_t const *>(inputs[2]);
                for (auto i : range0_(info.blockSize / value)) {
                    std::memcpy(defaultValue.data() + i * value, constValue, value);
                }
            }
            std::for_each_n(std::execution::par_unseq,
                            natural_t(0), info.blockCount,
                            [=, &info](auto i) {
                                long rem = i, j = 0;
                                bool flag = false;
                                for (auto const &dim : info.dims) {
                                    auto pos = rem / dim.strideO - dim.padS;
                                    if (pos < 0 || pos >= dim.dimI) {
                                        flag = true;
                                        break;
                                    }
                                    j += pos * dim.strideI;
                                    rem %= dim.strideO;
                                }
                                if (flag) {
                                    std::memcpy(dst + i * info.blockSize, defaultValue.data(), info.blockSize);
                                } else {
                                    std::memcpy(dst + i * info.blockSize, src + j * info.blockSize, info.blockSize);
                                }
                            });
        };
    }

}// namespace refactor::kernel
