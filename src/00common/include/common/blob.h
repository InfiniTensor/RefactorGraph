#ifndef COMMON_BLOB_H
#define COMMON_BLOB_H

#include "data_type.h"
#include <chrono>
#include <cstddef>

namespace refactor::common {

    /// @brief 内存块。
    struct Blob {
        /// @brief ! NOTICE 指针必须非空。
        void *ptr;
        DataType dataType;
        size_t size, stamp;

        explicit Blob(size_t);
        Blob(Blob const &) = delete;
        Blob(Blob &&) = delete;

        ~Blob();

        size_t hash() const;
    };

}// namespace refactor::common

#endif// COMMON_BLOB_H
