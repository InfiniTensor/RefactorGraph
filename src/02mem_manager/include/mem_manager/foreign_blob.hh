#ifndef MEM_MANAGER_FOREIGN_BLOB_H
#define MEM_MANAGER_FOREIGN_BLOB_H

#include "mem_functions.h"
#include <memory>

namespace refactor::mem_manager {

    /// @brief 显存内存块。
    class ForeignBlob {
        MemFunctions _memFunctions;

        /// @brief ! NOTICE 指针必须非空。
        void *_ptr;

        ForeignBlob(MemFunctions, size_t bytes);

    public:
        ForeignBlob(ForeignBlob const &) = delete;
        ForeignBlob(ForeignBlob &&) = delete;
        ~ForeignBlob();

        static std::shared_ptr<ForeignBlob> share(MemFunctions, size_t bytes);

        operator void const *() const noexcept;
        operator void *() noexcept;

        void copyIn(void const *, size_t);
        void copyOut(void *, size_t) const;
        void copyFrom(ForeignBlob const &, size_t);
        void copyTo(ForeignBlob &, size_t) const;
    };

    using SharedForeignBlob = std::shared_ptr<ForeignBlob>;

}// namespace refactor::mem_manager

#endif// MEM_MANAGER_FOREIGN_BLOB_H
