#ifndef MEM_MANAGER_BLOB_H
#define MEM_MANAGER_BLOB_H

#include "mem_functions.h"
#include <memory>

namespace refactor::mem_manager {

    /// @brief 一次初始化的内存块。
    class Blob {
        /// @brief ! NOTICE 指针必须非空。
        void *ptr;

        explicit Blob(size_t);

    public:
        Blob(Blob const &) = delete;
        Blob(Blob &&) = delete;
        ~Blob();

        static std::pair<std::shared_ptr<Blob>, void *> share(size_t);
        void const *operator*() const noexcept { return ptr; }
        template<class T> T const *get() const noexcept {
            return reinterpret_cast<T const *>(ptr);
        }
    };


    /// @brief 显存内存块。
    /// @tparam TMemFunctions 操作显存的函数组。
    template<class TMemFunctions>
    class ForeignBlob {
        /// @brief ! NOTICE 指针必须非空。
        void *ptr;

        explicit ForeignBlob(size_t bytes)
            : ptr(TMemFunctions::malloc(bytes)) {}

    public:
        ForeignBlob(ForeignBlob const &) = delete;
        ForeignBlob(ForeignBlob &&) = delete;
        ~ForeignBlob() { TMemFunctions::free(ptr); }

        static std::pair<std::shared_ptr<ForeignBlob>, void *> share(size_t bytes) {
            auto blob = std::make_shared<ForeignBlob>(bytes);
            auto ptr = blob->ptr;
            return {std::move(blob), ptr};
        }
        void const *operator*() const noexcept { return ptr; }
        void *operator*() noexcept { return ptr; }
    };

}// namespace refactor::mem_manager

#endif// MEM_MANAGER_BLOB_H
