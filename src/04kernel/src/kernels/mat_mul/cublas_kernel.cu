#include "../../utilities/cuda/cublas_context.hh"
#include "cublas_kernel.hh"

namespace refactor::kernel {
    using namespace runtime;
    using namespace cublas;

    auto MatMulCublas::lower() const noexcept -> Routine {
        return [](Resources &res, void const **inputs, void **outputs) {
            // fetch cublas handle from resources
            auto handle = res.fetchOrStore<CublasContext>()->handle;
        };
    }

}// namespace refactor::kernel
