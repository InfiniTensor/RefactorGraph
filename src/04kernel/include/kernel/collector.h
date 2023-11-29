#ifndef KERNEL_CANDIDATE_H
#define KERNEL_CANDIDATE_H

#include "hardware/device.h"
#include "kernel.h"
#include "tensor.h"

namespace refactor::kernel {

    class InfoCollector {
    protected:
        hardware::Device::Type _target;
        constexpr explicit InfoCollector(decltype(_target) target)
            : _target(target) {}

    public:
        virtual ~InfoCollector() = default;
        virtual std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const = 0;
    };

    using CollectorBox = std::unique_ptr<InfoCollector>;

}// namespace refactor::kernel

#endif// KERNEL_CANDIDATE_H
