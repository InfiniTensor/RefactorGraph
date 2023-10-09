#include "common/error_handler.h"
#include "cudnn_context.hh"
#include <cudnn.h>

namespace refactor::kernel::cuda {

    class CudnnContext::__Implement {
        cudnnHandle_t handle;

    public:
        __Implement() {
            if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
                RUNTIME_ERROR("Failed to create cudnn handle");
            }
        }
        ~__Implement() {
            if (cudnnDestroy(handle) != CUDNN_STATUS_SUCCESS) {
                RUNTIME_ERROR("Failed to destroy cudnn handle");
            }
        }

        __Implement(const __Implement &) = delete;
        __Implement &operator=(const __Implement &) = delete;
    };

    CudnnContext::CudnnContext() noexcept
        : runtime::Resource(), _impl(new __Implement) {}
    CudnnContext::~CudnnContext() noexcept {
        delete _impl;
        _impl = nullptr;
    }

    auto CudnnContext::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto CudnnContext::resourceTypeId() const noexcept -> size_t { return typeId(); }
    auto CudnnContext::description() const noexcept -> std::string_view {
        return "CudnnContext";
    }

}// namespace refactor::kernel::cuda
