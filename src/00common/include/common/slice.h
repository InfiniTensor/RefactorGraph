#ifndef SLICE_H
#define SLICE_H

namespace refactor::common {
    template<class t>
    struct slice_t {
        t const *begin_, *end_;

        using Iterator = t const *;

        bool empty() const { return end_ == begin_; }
        size_t size() const { return end_ - begin_; }
        t const &at(size_t i) const {
            ASSERT(i < size(), "Index out of range");
            return operator[](i);
        }
        t const &operator[](int i) const { return begin_[i]; }
        Iterator begin() const { return begin_; }
        Iterator end() const { return end_; }
    };
}// namespace refactor::common

#endif// SLICE_H
