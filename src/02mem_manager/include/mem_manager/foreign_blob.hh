#ifndef MEM_MANAGER_FOREIGN_BLOB_H
#define MEM_MANAGER_FOREIGN_BLOB_H

#include "mem_functions.h"
#include <memory>

namespace refactor::mem_manager {

    /// @brief 显存内存块。
    class ForeignBlob {
        MemFunctions const &_memFunctions;

        /// @brief ! NOTICE 指针必须非空。
        void *_ptr;

        ForeignBlob(MemFunctions const &, size_t bytes);

    public:
        ForeignBlob(ForeignBlob const &) = delete;
        ForeignBlob(ForeignBlob &&) = delete;
        ~ForeignBlob();

        static std::shared_ptr<ForeignBlob> share(MemFunctions const &, size_t bytes);
        void const *operator*() const noexcept;
        void *operator*() noexcept;
    };

}// namespace refactor::mem_manager

#endif// MEM_MANAGER_FOREIGN_BLOB_H
