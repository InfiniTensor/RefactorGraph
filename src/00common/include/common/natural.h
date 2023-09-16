#ifndef NATURAL_H
#define NATURAL_H

#include <iterator>

namespace refactor::common {

    template<class t = size_t>
    class natural_t : public std::iterator<std::input_iterator_tag, t> {
        size_t _i;

    public:
        natural_t(t val) : _i(val) {}
        bool operator==(natural_t const &rhs) const { return _i == rhs._i; }
        bool operator!=(natural_t const &rhs) const { return _i != rhs._i; }
        bool operator<(natural_t const &rhs) const { return _i < rhs._i; }
        bool operator>(natural_t const &rhs) const { return _i > rhs._i; }
        bool operator<=(natural_t const &rhs) const { return _i <= rhs._i; }
        bool operator>=(natural_t const &rhs) const { return _i >= rhs._i; }
        natural_t &operator++() {
            ++_i;
            return *this;
        }
        natural_t operator++(int) {
            auto ans = *this;
            operator++();
            return ans;
        }
        t operator*() const {
            return _i;
        }
    };

}// namespace refactor::common

#endif// NATURAL_H
