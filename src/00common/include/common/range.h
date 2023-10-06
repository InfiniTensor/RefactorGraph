#ifndef RANGE_H
#define RANGE_H

#include "error_handler.h"
#include "natural.h"

namespace refactor::common {
    template<class t> struct range_t;
    template<class t> struct rev_range_t;

    template<class t = size_t>
    struct range_t {
        t begin_, end_;

        using Iterator = common::natural_t<t>;

        bool empty() const noexcept { return end_ == begin_; }
        size_t size() const noexcept { return end_ - begin_; }
        t at(size_t i) const noexcept {
            ASSERT(i < size(), "Index out of range");
            return operator[](i);
        }
        t operator[](size_t i) const noexcept { return begin_ + i; }
        Iterator begin() const noexcept { return begin_; }
        Iterator end() const noexcept { return end_; }
        rev_range_t<t> rev() const noexcept { return {end_, begin_}; }
    };

    template<class t = size_t>
    struct rev_range_t {
        t begin_, end_;

        using Iterator = common::rev_natural_t<t>;

        bool empty() const noexcept { return end_ == begin_; }
        size_t size() const noexcept { return end_ - begin_; }
        t at(size_t i) const {
            ASSERT(i < size(), "Index out of range");
            return operator[](i);
        }
        t operator[](size_t i) const noexcept { return begin_ + i; }
        Iterator begin() const noexcept { return begin_; }
        Iterator end() const noexcept { return end_; }
    };

    template<class t = size_t> range_t<t> range0_(t end) {
        ASSERT(end >= 0, "end must be greater than 0");
        return {0, end};
    }
    template<class t = size_t> range_t<t> range(t begin, t end) {
        ASSERT(begin <= end, "begin must be less than end");
        return {begin, end};
    }
}// namespace refactor::common


#endif// RANGE_H
