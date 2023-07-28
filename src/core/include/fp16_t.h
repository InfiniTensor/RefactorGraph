#ifndef FP16_H
#define FP16_H

#include <array>
#include <cstdint>
#include <fmt/core.h>

namespace proj_namespace::core {

    class fp16_t final {
        uint16_t code;

        constexpr static uint16_t mask_low(int bits) {
            return (1 << bits) - 1;
        }

        constexpr static uint16_t mask_high(int bits) {
            return ~(mask_low(bits));
        }

        constexpr static uint16_t from_f32(float val) {
            union f32 {
                float f32;
                uint32_t u32;
            } x{val};
            return (static_cast<uint16_t>(x.u32 >> 16) & mask_high(15)) |
                   (((static_cast<uint16_t>(x.u32 >> 23) - 127 + 15) & mask_low(5)) << 10) |
                   (static_cast<uint16_t>(x.u32 >> 13) & mask_low(10));
        }

    public:
        constexpr fp16_t() noexcept : code(code) {}
        constexpr fp16_t(uint16_t code) noexcept : code(code) {}
        constexpr fp16_t(float value) noexcept : code(from_f32(value)) {}
        constexpr fp16_t(fp16_t const &) noexcept = default;
        constexpr fp16_t(fp16_t &&) noexcept = default;

        constexpr uint16_t as_code() const {
            return code;
        }

        constexpr float to_f32() const {
            union f32 {
                uint32_t u32;
                float f32;
            } ans{0};
            ans.u32 = ((code << 16) & mask_high(31)) |
                      ((((code >> 10) & mask_low(5)) - 15 + 127) << 23) |
                      ((code & mask_low(10)) << 13);
            return ans.f32;
        }

        constexpr std::array<char, 38> format() const {
            std::array<char, 38> ans{"0'00000'0000000000\n+ 2^-15x1.        "};
            ans[0] += (code >> 15);
            ans[0 + 19] = ans[0] == '0' ? '+' : '-';

            for (auto i = 0; i < 5; ++i) {
                ans[i + 2] += (code >> (14 - i)) & 1;
            }
            int exp = ((code >> 10) & mask_low(5)) - 15;
            ans[4 + 19] = exp > 0 ? '+' : '-';
            ans[5 + 19] = '0' + std::abs(exp) / 10;
            ans[6 + 19] = '0' + std::abs(exp) % 10;

            for (auto i = 0; i < 10; ++i) {
                ans[i + 8] += (code >> (9 - i)) & 1;
            }
            sprintf(ans.data() + 8 + 19, "%.8f", (static_cast<float>(code & mask_low(10)) / 1024.0f) + 1);

            return ans;
        }

        std::string to_string() const {
            return std::string(format().data());
        }
    };

}// namespace proj_namespace::core

#endif// FP16_H
