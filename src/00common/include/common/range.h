#ifndef RANGE_H
#define RANGE_H

#include "natural.h"

namespace refactor::common {
    template<class t = size_t>
    struct range_t {
        t begin_, end_;

        using Iterator = common::natural_t<t>;

        bool empty() const { return end_ == begin_; }
        size_t size() const { return end_ - begin_; }
        t at(size_t i) const {
            ASSERT(i < size(), "Index out of range");
            return operator[](i);
        }
        t operator[](size_t i) const { return begin_ + i; }
        Iterator begin() const { return begin_; }
        Iterator end() const { return end_; }
    };

    template<class t = size_t> range_t<t> range0_(t end) { return {0, end}; }
    template<class t = size_t> range_t<t> range(t begin, t end) { return {begin, end}; }
}// namespace refactor::common


#endif// RANGE_H
