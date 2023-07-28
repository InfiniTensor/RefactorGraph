#ifndef FP16_H
#define FP16_H

#include <cstdint>
#include <fmt/core.h>

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

    std::string to_string() const {
        char buf_line[2][19]{"0'00000'0000000000", "+ 2^-15x1.        "};
        buf_line[0][0] += (code >> 15);
        buf_line[1][0] = buf_line[0][0] == '0' ? '+' : '-';

        for (auto i = 0; i < 5; ++i) {
            buf_line[0][i + 2] += (code >> (14 - i)) & 1;
        }
        int exp = ((code >> 10) & mask_low(5)) - 15;
        buf_line[1][4] = exp > 0 ? '+' : '-';
        buf_line[1][5] = '0' + std::abs(exp) / 10;
        buf_line[1][6] = '0' + std::abs(exp) % 10;

        for (auto i = 0; i < 5; ++i) {
            buf_line[0][i + 8] += (code >> (9 - i)) & 1;
        }
        sprintf(buf_line[1] + 8, "%.8f", (static_cast<float>(code & mask_low(10)) / 1024.0f) + 1);

        return std::string(buf_line[0]) +
               "\n" + std::string(buf_line[1]);
    }
};

#endif// FP16_H
