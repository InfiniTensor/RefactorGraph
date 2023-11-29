#include "computation/operators/einsum.h"

namespace refactor::computation {
    using Op = Einsum;

    EinsteinNotation::EinsteinNotation(std::string_view equation)
        : _items(), _indices() {
        auto implicit = true;
        // 整理并记录序号以实现随机访问
        for (size_t i = 0; i < equation.size();) {
            switch (equation[i]) {
                case ' ':
                    ++i;
                    break;

                case ',':
                    ++i;
                    ASSERT(implicit, "");
                    _items.push_back(_indices.size());
                    break;

                case '-':
                    ASSERT(i + 1 < equation.size(), "");
                    ASSERT(equation[i + 1] == '>', "");
                    i += 2;
                    implicit = false;
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
        if (implicit) {
            _items.push_back(_indices.size());
            std::vector<bool> appear(26 * 2, false);
            dim_t begin = 0;
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
            for (auto i : range(_items.back(), static_cast<dim_t>(_indices.size()))) {
                noSum.insert(_indices[i]);
            }
        }
        // 标记哑指标
        for (auto &c : _indices) {
            if (noSum.contains(c)) {
                c <<= 1;
            } else {
                c <<= 1;
                c |= 1;
            }
        }
    }

    size_t EinsteinNotation::items() const noexcept {
        return _items.size();
    }

    std::string EinsteinNotation::indices(size_t i) const noexcept {
        std::string ans;
        for (auto j : range(i == 0 ? 0 : _items[i - 1], _items[i])) {
            ans += static_cast<char>(_indices[j] >> 1);
        }
        return ans;
    }

    std::string EinsteinNotation::outputIndices() const noexcept {
        std::string ans;
        for (auto i : range(_items.back(), static_cast<dim_t>(_indices.size()))) {
            ans += static_cast<char>(_indices[i] >> 1);
        }
        return ans;
    }

    std::string EinsteinNotation::sumIndices() const noexcept {
        std::vector<bool> sum(26, false);
        for (auto i : range0_(_items.back())) {
            if (_indices[i] & 1) {
                sum[(_indices[i] >> 1) - 'a'] = true;
            }
        }
        std::string ans;
        for (auto i : range0_(sum.size())) {
            if (sum[i]) {
                ans += static_cast<char>(i + 'a');
            }
        }
        return ans;
    }

    std::string EinsteinNotation::toString() const noexcept {
        std::string ans;
        ans += static_cast<char>('A' + _items.size());
        ans += '_';
        ans += outputIndices();
        ans += " = Σ";
        ans += sumIndices();
        ans += ": ";
        dim_t begin = 0;
        for (auto i : range0_(_items.size())) {
            ans += static_cast<char>('A' + i);
            ans += '_';
            for (auto j : range(begin, _items[i])) {
                if (_indices[j] & 1) {
                    ans += '(';
                    ans += static_cast<char>(_indices[j] >> 1);
                    ans += ')';
                } else {
                    ans += static_cast<char>(_indices[j] >> 1);
                }
            }
            ans += " * ";
            begin = _items[i];
        }
        return ans.substr(0, ans.size() - 3);
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
