#ifndef NATURAL_H
#define NATURAL_H

#include <iterator>

namespace refactor {

    template<class t = size_t>
    class natural_t : public std::iterator<std::input_iterator_tag, t> {
        size_t _i;

    public:
        natural_t(t val) noexcept : _i(val) {}
        bool operator==(natural_t const &rhs) const noexcept { return _i == rhs._i; }
        bool operator!=(natural_t const &rhs) const noexcept { return _i != rhs._i; }
        bool operator<(natural_t const &rhs) const noexcept { return _i < rhs._i; }
        bool operator>(natural_t const &rhs) const noexcept { return _i > rhs._i; }
        bool operator<=(natural_t const &rhs) const noexcept { return _i <= rhs._i; }
        bool operator>=(natural_t const &rhs) const noexcept { return _i >= rhs._i; }
        natural_t &operator++() noexcept {
            ++_i;
            return *this;
        }
        natural_t operator++(int) noexcept {
            auto ans = *this;
            operator++();
            return ans;
        }
        t operator*() const noexcept {
            return _i;
        }
    };

    template<class t = size_t>
    class rev_natural_t : public std::iterator<std::input_iterator_tag, t> {
        size_t _i;

    public:
        rev_natural_t(t val) noexcept : _i(val - 1) {}
        bool operator==(rev_natural_t const &rhs) const noexcept { return _i == rhs._i; }
        bool operator!=(rev_natural_t const &rhs) const noexcept { return _i != rhs._i; }
        bool operator<(rev_natural_t const &rhs) const noexcept { return _i > rhs._i; }
        bool operator>(rev_natural_t const &rhs) const noexcept { return _i < rhs._i; }
        bool operator<=(rev_natural_t const &rhs) const noexcept { return _i >= rhs._i; }
        bool operator>=(rev_natural_t const &rhs) const noexcept { return _i <= rhs._i; }
        rev_natural_t &operator++() noexcept {
            --_i;
            return *this;
        }
        rev_natural_t operator++(int) noexcept {
            auto ans = *this;
            operator++();
            return ans;
        }
        t operator*() const noexcept {
            return _i;
        }
    };

}// namespace refactor

#endif// NATURAL_H
