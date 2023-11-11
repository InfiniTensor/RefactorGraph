#ifndef COMMON_TYPES
#define COMMON_TYPES

#include "common/bf16_t.h"
#include "common/data_type.h"
#include "common/error_handler.h"
#include "common/fp16_t.h"
#include "common/range.h"
#include "common/rc.hpp"
#include "common/slice.h"
#include <absl/container/inlined_vector.h>
#include <memory>
#include <sstream>

namespace refactor {
    /// @brief 用于表示维度/形状的差的数值，主要是一些属性。
    using ddim_t = int16_t;
    /// @brief 用于表示形状的数值。
    using dim_t = uint32_t;
    /// @brief 用于表示对象的数量。
    using count_t = uint32_t;

    template<class T> using Arc = std::shared_ptr<T>;

    template<class Container> std::string vec2str(Container const &vec) {
        std::stringstream ss;
        ss << "[ ";
        for (auto d : vec) {
            ss << d << ' ';
        }
        ss << ']';
        return ss.str();
    }

}// namespace refactor

#endif// COMMON_TYPES
