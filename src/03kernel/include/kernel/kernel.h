#ifndef KERNEL_KERNEL_H
#define KERNEL_KERNEL_H

#include <absl/container/inlined_vector.h>
#include <functional>
#include <string_view>

namespace refactor::kernel {

    using Addresses = absl::InlinedVector<void *, 2>;

    class Kernel {
    public:
        virtual std::string_view description() const = 0;
        virtual std::function<void(Addresses, Addresses)> lower() const = 0;
    };

}// namespace refactor::kernel

#endif// KERNEL_KERNEL_H
