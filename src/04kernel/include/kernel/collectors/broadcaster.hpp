#ifndef KERNEL_BROADCASTER_HPP
#define KERNEL_BROADCASTER_HPP

#include "../tensor.h"

namespace refactor::kernel {

    template<class T, size_t N>
    class Broadcaster {
        constexpr size_t countT() const {
            auto bits = sizeof(T) * 8;
            return (N + bits - 1) / bits;
        }

        struct Dimension {
            T size;
            T steps[countT()];
        };
        std::vector<Dimension> _dims;

    public:
        Broadcaster(std::vector<std::reference_wrapper<Shape const>> const inputs) {
        }
    };

}// namespace refactor::kernel

#endif// KERNEL_BROADCASTER_HPP
