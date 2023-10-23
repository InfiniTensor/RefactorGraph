#ifndef MEM_MANAGER_FOREIGN_BLOB_H
#define MEM_MANAGER_FOREIGN_BLOB_H

#include "common.h"
#include "mem_manager.hh"

namespace refactor::mem_manager {

    /// @brief 显存内存块。
    class ForeignBlob {
        Arc<MemManager> _memManager;

        /// @brief ! NOTICE 指针必须非空。
        void *_ptr;

        ForeignBlob(Arc<MemManager>, size_t bytes);

    public:
        ForeignBlob(ForeignBlob const &) = delete;
        ForeignBlob(ForeignBlob &&) = delete;
        ~ForeignBlob();

        static Arc<ForeignBlob> share(Arc<MemManager>, size_t bytes);

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
