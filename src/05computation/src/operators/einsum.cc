#include "computation/operators/einsum.h"

namespace refactor::computation {
    using Op = Einsum;

    EinsteinNotation::EinsteinNotation(std::string_view equation)
        : _implicit(true), _items(), _indices() {
        // 整理并记录序号以实现随机访问
        for (size_t i = 0; i < equation.size();) {
            switch (equation[i]) {
                case ' ':
                    ++i;
                    break;

                case ',':
                    ++i;
                    ASSERT(_implicit, "");
                    _items.push_back(_indices.size());
                    break;

                case '-':
                    ASSERT(i + 1 < equation.size(), "");
                    ASSERT(equation[i + 1] == '>', "");
                    i += 2;
                    _implicit = false;
                    _items.push_back(_indices.size());
                    break;

                case '.':
                    ASSERT(i + 2 < equation.size(), "");
                    ASSERT(equation[i + 1] == '.' && equation[i + 2] == '.', "");
                    i += 3;
                    _indices.push_back('.');
                    break;

                default:
                    if ('a' <= equation[i] && equation[i] <= 'z') {
                        _indices.push_back(equation[i++]);
                    } else {
                        UNREACHABLE();
                    }
                    break;
            }
        }
        // 分析自由度
        std::unordered_set<uint8_t> noSum{'.'};
        if (_implicit) {
            _items.push_back(_indices.size());
            std::vector<bool> appear(26 * 2, false);
            uint_lv2 begin = 0;
            for (auto i : range0_(_items.size())) {
                for (auto j : range(begin, _items[i])) {
                    if (_indices[j] == '.') {
                        continue;
                    }
                    auto c = _indices[j] - 'a';
                    if (!appear[c + c]) {
                        appear[c + c] = true;
                    } else {
                        appear[c + c + 1] = true;
                    }
                }
                begin = _items[i];
            }
            for (auto i = 0; i < 26; ++i) {
                if (appear[i + i] && !appear[i + i + 1]) {
                    auto c = static_cast<uint8_t>(i + 'a');
                    noSum.insert(c);
                    _indices.push_back(c);
                }
            }
        } else {
            for (auto i : range(_items.back(), static_cast<uint_lv2>(_indices.size()))) {
                noSum.insert(_indices[i]);
            }
        }
        // 标记哑变量
        for (auto &c : _indices) {
            if (noSum.find(c) != noSum.end()) {
                c <<= 1;
            } else {
                c <<= 1;
                c |= 1;
            }
        }
    }

    std::string EinsteinNotation::toString() const noexcept {
        std::string msg;
        msg += static_cast<char>('A' + _items.size());
        msg += '_';
        for (auto i : range(_items.back(), static_cast<uint_lv2>(_indices.size()))) {
            msg += static_cast<char>(_indices[i] >> 1);
        }
        msg += " = Σ";
        std::vector<bool> sum(26, false);
        for (auto i : range0_(_items.back())) {
            if (_indices[i] & 1) {
                sum[(_indices[i] >> 1) - 'a'] = true;
            }
        }
        for (auto i : range0_(sum.size())) {
            if (sum[i]) {
                msg += '_';
                msg += static_cast<char>(i + 'a');
            }
        }
        msg += ' ';
        uint_lv2 begin = 0;
        for (auto i : range0_(_items.size())) {
            msg += static_cast<char>('A' + i);
            msg += '_';
            for (auto j : range(begin, _items[i])) {
                if (_indices[j] & 1) {
                    msg += '(';
                    msg += static_cast<char>(_indices[j] >> 1);
                    msg += ')';
                } else {
                    msg += static_cast<char>(_indices[j] >> 1);
                }
            }
            msg += " * ";
            begin = _items[i];
        }
        return msg.substr(0, msg.size() - 3);
    }

    Op::Einsum(std::string_view equation) noexcept
        : Operator(), notation(equation) {}

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "Einsum"; }

}// namespace refactor::computation
