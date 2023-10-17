#ifndef COMPUTATION_EINSUM_H
#define COMPUTATION_EINSUM_H

#include "../operator.h"

namespace refactor::computation {

    class EinsteinNotation {
        bool _implicit;
        std::vector<uint_lv2> _items;
        std::vector<uint8_t> _indices;

    public:
        explicit EinsteinNotation(std::string_view equation);
        std::string toString() const noexcept;
    };

    struct Einsum final : public Operator {
        EinsteinNotation notation;

        explicit Einsum(std::string_view) noexcept;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_EINSUM_H
