#ifndef KERNEL_KERNEL_H
#define KERNEL_KERNEL_H

#include "runtime/stream.h"
#include <string_view>

namespace refactor::kernel {
    using runtime::Resources;
    using runtime::Routine;
    using RoutineWorkspace = runtime::Node;

    class Kernel {
    public:
        virtual ~Kernel() = default;
        virtual size_t kernelTypeId() const = 0;
        virtual std::string_view description() const = 0;
        virtual RoutineWorkspace lower(Resources &) const;

        template<class T, class... Args>
        bool is(Args &&...args) const noexcept {
            return this->kernelTypeId() == T::typeId(std::forward<Args>(args)...);
        }
    };

    using KernelBox = std::unique_ptr<Kernel>;

}// namespace refactor::kernel

#endif// KERNEL_KERNEL_H
