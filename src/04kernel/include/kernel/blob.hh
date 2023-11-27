#ifndef KERNEL_BLOB_H
#define KERNEL_BLOB_H

#include "common.h"

namespace refactor::kernel {

    /// @brief 一次初始化的内存块。
    class Blob {
        /// @brief ! NOTICE 指针必须非空。
        void *_ptr;

        explicit Blob(size_t);

    public:
        Blob(Blob const &) = delete;
        Blob(Blob &&) = delete;
        ~Blob();

        static std::pair<Arc<Blob>, void *> share(size_t);
        operator void const *() const noexcept;
        template<class T> T const *get() const noexcept {
            return reinterpret_cast<T const *>(_ptr);
        }
    };

}// namespace refactor::kernel

#endif// KERNEL_BLOB_H
