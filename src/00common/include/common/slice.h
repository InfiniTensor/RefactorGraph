#ifndef SLICE_H
#define SLICE_H

#include <cstddef>

namespace refactor {
    template<class t>
    struct slice_t {
        t const *begin_, *end_;

        using Iterator = t const *;

        bool empty() const noexcept { return end_ == begin_; }
        size_t size() const noexcept { return end_ - begin_; }
        t const &at(size_t i) const {
            ASSERT(i < size(), "Index out of range");
            return operator[](i);
        }
        t const &operator[](int i) const noexcept { return begin_[i]; }
        Iterator begin() const noexcept { return begin_; }
        Iterator end() const noexcept { return end_; }
    };

    template<class t> slice_t<t> slice(t const *begin, t const *end) noexcept { return {begin, end}; }
    template<class t> slice_t<t> slice(t const *begin, size_t size) noexcept { return {begin, begin + size}; }
}// namespace refactor

#endif// SLICE_H
