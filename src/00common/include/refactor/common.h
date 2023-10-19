#ifndef COMMON_TYPES
#define COMMON_TYPES

#include "bf16_t.h"
#include "data_type.h"
#include "error_handler.h"
#include "fp16_t.h"
#include "range.h"
#include "rc.hpp"
#include "slice.h"
#include <memory>

namespace refactor {
    // 方便按“级别”定义整型数。

    using sint_lv0 = int8_t;
    using sint_lv1 = int16_t;
    using sint_lv2 = int32_t;
    using sint_lv3 = int64_t;

    using uint_lv0 = uint8_t;
    using uint_lv1 = uint16_t;
    using uint_lv2 = uint32_t;
    using uint_lv3 = uint64_t;

    using sint_min = sint_lv0;
    using sint_max = sint_lv3;
    using uint_min = uint_lv0;
    using uint_max = uint_lv3;

    template<class T> using Arc = std::shared_ptr<T>;

}// namespace refactor

#endif// COMMON_TYPES
