#include "common/error_handler.h"
#include "cudnn_context.hh"
#include <cudnn.h>

namespace refactor::kernel::cudnn {

    CudnnContext::CudnnContext() noexcept
        : runtime::Resource() {
        cudnnHandle_t handle_;
        if (cudnnCreate(&handle_) != CUDNN_STATUS_SUCCESS) {
            RUNTIME_ERROR("Failed to create cudnn handle");
        }
        handle = handle_;
    }
    CudnnContext::~CudnnContext() noexcept {
        auto handle_ = std::any_cast<cudnnHandle_t>(handle);
        if (cudnnDestroy(handle_) != CUDNN_STATUS_SUCCESS) {
            RUNTIME_ERROR("Failed to destroy cudnn handle");
        }
    }

    auto CudnnContext::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto CudnnContext::build() noexcept -> runtime::ResourceBox {
        return std::make_unique<CudnnContext>();
    }

    auto CudnnContext::resourceTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto CudnnContext::description() const noexcept -> std::string_view {
        return "CudnnContext";
    }

}// namespace refactor::kernel::cudnn
