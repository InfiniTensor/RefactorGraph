#ifndef FP16_H
#define FP16_H

#include <array>
#include <cstdint>
#include <fmt/core.h>

namespace refactor {

    class fp16_t final {
        uint16_t code;

        const static uint16_t MASK_SIGN16 = 0b1'00000'0000000000;
        const static uint16_t MASK_EXP_16 = 0b0'11111'0000000000;
        const static uint16_t MASK_TAIL16 = 0b0'00000'1111111111;

        constexpr static uint16_t mask_low(int bits) noexcept {
            return (1 << bits) - 1;
        }

        constexpr static uint16_t from_f32(float val) noexcept {
            union {
                float f32;
                uint32_t u32;
            } x{val};
            return (static_cast<uint16_t>(x.u32 >> 16) & MASK_SIGN16) |
                   (((static_cast<uint16_t>(x.u32 >> 23) - 127 + 15) & mask_low(5)) << 10) |
                   (static_cast<uint16_t>(x.u32 >> 13) & MASK_TAIL16);
        }

    public:
        static const fp16_t ZERO, ONE, INF;

        constexpr fp16_t() noexcept : code(from_f32(0.0)) {}
        constexpr fp16_t(uint16_t code) noexcept : code(code) {}
        constexpr fp16_t(float value) noexcept : code(from_f32(value)) {}
        constexpr fp16_t(fp16_t const &) noexcept = default;
        constexpr fp16_t(fp16_t &&) noexcept = default;

        constexpr uint16_t as_code() const noexcept {
            return code;
        }

        constexpr float to_f32() const noexcept {
            union {
                uint32_t u32;
                float f32;
            } ans{0};
            ans.u32 = ((code << 16) & (1 << 31)) |
                      ((((code >> 10) & mask_low(5)) - 15 + 127) << 23) |
                      ((code & MASK_TAIL16) << 13);
            return ans.f32;
        }

        constexpr bool is_inf() const noexcept {
            return MASK_EXP_16 == (code & MASK_EXP_16) && 0 == (code & MASK_TAIL16);
        }

        constexpr bool is_nan() const noexcept {
            return MASK_EXP_16 == (code & MASK_EXP_16) && 0 != (code & MASK_TAIL16);
        }

        constexpr fp16_t operator-() const noexcept {
            return (uint16_t) (code ^ (code | MASK_SIGN16));
        }

        constexpr bool operator==(fp16_t const &others) const noexcept {
            return code == others.code && !is_nan();
        }

        constexpr bool operator!=(fp16_t const &others) const noexcept {
            return !operator==(others);
        }

        constexpr std::array<char, 38> format() const noexcept {
            // 将 fp16 格式化字符串保存到栈上的内存块上。
            std::array<char, 38> ans{"0'00000'0000000000\n+ 2^-15x1.        "};
            ans[0] += (code >> 15);
            ans[0 + 19] = ans[0] == '0' ? '+' : '-';

            for (auto i = 0; i < 5; ++i) {
                ans[i + 2] += (code >> (14 - i)) & 1;
            }
            int exp = ((code & MASK_EXP_16) >> 10) - 15;
            ans[4 + 19] = exp > 0 ? '+' : '-';
            ans[5 + 19] = '0' + std::abs(exp) / 10;
            ans[6 + 19] = '0' + std::abs(exp) % 10;

            for (auto i = 0; i < 10; ++i) {
                ans[i + 8] += (code >> (9 - i)) & 1;
            }
            sprintf(ans.data() + 8 + 19, "%.8f", (static_cast<float>(code & MASK_TAIL16) / 1024.0f) + 1);

            return ans;
        }

        std::string to_string() const noexcept {
            return std::string(format().data());
        }
    };


    inline const fp16_t fp16_t::ZERO = fp16_t(0.0f);
    inline const fp16_t fp16_t::ONE = fp16_t(1.0f);
    inline const fp16_t fp16_t::INF = fp16_t((uint16_t) 0b0'11111'0000000000);
    // inline const fp16_t fp16_t::NAN = fp16_t((uint16_t) 0b0'11111'1000000000);

}// namespace refactor

#endif// FP16_H
