#ifndef MEM_MANAGER_BLOB_H
#define MEM_MANAGER_BLOB_H

#include <memory>

namespace refactor::mem_manager {

    /// @brief 一次初始化的内存块。
    class Blob {
        /// @brief ! NOTICE 指针必须非空。
        void *_ptr;

        explicit Blob(size_t);

    public:
        Blob(Blob const &) = delete;
        Blob(Blob &&) = delete;
        ~Blob();

        static std::pair<std::shared_ptr<Blob>, void *> share(size_t);
        operator void const *() const noexcept;
        operator void *() noexcept;
        template<class T> T const *get() const noexcept {
            return reinterpret_cast<T const *>(_ptr);
        }
    };

    using SharedBlob = std::shared_ptr<Blob>;

}// namespace refactor::mem_manager

#endif// MEM_MANAGER_BLOB_H
