#ifndef COMMON_BLOB_H
#define COMMON_BLOB_H

#include <cstddef>

namespace refactor::common {

    /// @brief 内存块。
    struct Blob {
        /// @brief ! NOTICE 指针必须非空。
        void *ptr;

        explicit Blob(size_t);
        Blob(Blob const &) = delete;
        Blob(Blob &&) = delete;

        ~Blob();
    };

}// namespace refactor::common

#endif// COMMON_BLOB_H
