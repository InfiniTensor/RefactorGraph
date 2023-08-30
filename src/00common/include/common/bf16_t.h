#ifndef BF16_H
#define BF16_H

#include <cmath>
#include <cstdint>
#include <string>

namespace refactor::common {

    class bf16_t final {
        uint16_t code;

        union converter {
            float f32;
            uint16_t u16[2];
        };

        constexpr static uint16_t MASK_SIGN16 = 0b1'00000'00000'00000;

    public:
        constexpr bf16_t() noexcept : code(converter{0.0}.u16[1]) {}
        constexpr bf16_t(uint16_t code) noexcept : code(code) {}
        constexpr bf16_t(float value) noexcept : code(converter{value}.u16[1]) {}
        constexpr bf16_t(bf16_t const &) noexcept = default;
        constexpr bf16_t(bf16_t &&) noexcept = default;

        constexpr uint16_t as_code() const {
            return code;
        }

        constexpr float to_f32() const {
            converter c{};
            c.u16[1] = code;
            c.u16[0] = 0;
            return c.f32;
        }

        constexpr bool is_inf() const {
            return std::isinf(to_f32());
        }

        constexpr bool is_nan() const {
            return std::isnan(to_f32());
        }

        constexpr bf16_t operator-() const {
            return (uint16_t) (code ^ (code | MASK_SIGN16));
        }

        constexpr bool operator==(bf16_t const &others) const {
            return code == others.code && !is_nan();
        }

        constexpr bool operator!=(bf16_t const &others) const {
            return !operator==(others);
        }

        std::string to_string() const {
            return std::to_string(to_f32());
        }
    };

}// namespace refactor::common

#endif// BF16_H
