#ifndef COMPUTATION_EINSUM_H
#define COMPUTATION_EINSUM_H

#include "../operator.h"

namespace refactor::computation {

    class EinsteinNotation {
        std::vector<uint_lv2> _items;
        std::vector<uint8_t> _indices;

    public:
        explicit EinsteinNotation(std::string_view equation);

        size_t items() const noexcept;
        std::string indices(size_t) const noexcept;
        std::string outputIndices() const noexcept;
        std::string sumIndices() const noexcept;
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
