#ifndef KERNEL_CNNL_CONTEXT_HH
#define KERNEL_CNNL_CONTEXT_HH

#include "runtime/resource.h"
#include <cnnl.h>
#include <cnrt.h>

namespace refactor::kernel::cnnl {

    struct CnnlContext final : public runtime::Resource {
        cnnlHandle_t handle;
        cnrtQueue_t queue;

        CnnlContext();
        ~CnnlContext();
        CnnlContext(CnnlContext const &) noexcept = delete;
        CnnlContext(CnnlContext &&) noexcept = delete;

        static size_t typeId() noexcept;
        static runtime::ResourceBox build();

        size_t resourceTypeId() const noexcept final;
        std::string_view description() const noexcept final;

        void copyFromCPU(void *dst, const void *src, size_t size);
        void queueSync();
    };

}// namespace refactor::kernel::cnnl

#endif// KERNEL_CNNL_CONTEXT_HH

