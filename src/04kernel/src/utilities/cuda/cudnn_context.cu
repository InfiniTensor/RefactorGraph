#include "cudnn_context.hh"
#include "cudnn_functions.h"

namespace refactor::kernel::cudnn {

    CudnnContext::CudnnContext() : runtime::Resource() {
        CUDNN_ASSERT(cudnnCreate(&handle));
    }
    CudnnContext::~CudnnContext() {
        CUDNN_ASSERT(cudnnDestroy(handle));
    }

    auto CudnnContext::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto CudnnContext::build() -> runtime::ResourceBox {
        return std::make_unique<CudnnContext>();
    }

    auto CudnnContext::resourceTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto CudnnContext::description() const noexcept -> std::string_view {
        return "CudnnContext";
    }

}// namespace refactor::kernel::cudnn
