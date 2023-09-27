#ifndef MEM_MANAGER_BLOB_H
#define MEM_MANAGER_BLOB_H

#include <cstddef>
#include <memory>

namespace refactor::mem_manager {

    /// @brief 内存块。
    class Blob {
        /// @brief ! NOTICE 指针必须非空。
        void *ptr;

        explicit Blob(size_t);

    public:
        Blob(Blob const &) = delete;
        Blob(Blob &&) = delete;
        ~Blob();

        static std::pair<std::shared_ptr<Blob>, void *> share(size_t);
        template<class T> T const *get() const {
            return reinterpret_cast<T const *>(ptr);
        }
    };

}// namespace refactor::mem_manager

#endif// MEM_MANAGER_BLOB_H
