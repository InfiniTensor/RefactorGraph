#ifdef USE_BANG

#include "cnnl_context.hh"
#include "cnnl_functions.h"

namespace refactor::kernel::cnnl {

    CnnlContext::CnnlContext() : runtime::Resource() {
        BANG_ASSERT(cnrtQueueCreate(&queue));
        CNNL_ASSERT(cnnlCreate(&handle));
        CNNL_ASSERT(cnnlSetQueue(handle, queue));
    }
    CnnlContext::~CnnlContext() {
        BANG_ASSERT(cnrtQueueDestroy(queue));
        CNNL_ASSERT(cnnlDestroy(handle));
    }

    auto CnnlContext::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto CnnlContext::build() -> runtime::ResourceBox {
        return std::make_unique<CnnlContext>();
    }

    auto CnnlContext::resourceTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto CnnlContext::description() const noexcept -> std::string_view {
        return "CnnlContext";
    }

}// namespace refactor::kernel::cnnl

#endif
