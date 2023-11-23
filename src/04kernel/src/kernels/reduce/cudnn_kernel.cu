#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "common.h"
#include "cudnn_kernel.hh"
#include "mem_manager/functions.h"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;

    auto ReduceCudnn::lower(Resources &res) const noexcept -> RoutineWorkspace {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x;
            cudnnTensorDescriptor_t y;
            cudnnReduceTensorDescriptor_t reduce;

            Descriptors() {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&y));
                CUDNN_ASSERT(cudnnCreateReduceTensorDescriptor(&reduce));
            }
            ~Descriptors() {
                // Destories in CUDA does not require sync.
                // But cuDNN does not state whether sync is required before destories.
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(y));
                CUDNN_ASSERT(cudnnDestroyReduceTensorDescriptor(reduce));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();

        std::vector<int> dimsI(shape.begin(), shape.end()), dimsO(shape.begin(), shape.end());
        for (auto axis : axes) {
            dimsO[axis] = 1;
        }

        auto cudnnDataType = cudnnDataTypeConvert(dataType);
        if (auto n = shape.size(); n <= 4) {
            int idims[4] = {1, 1, 1, 1}, odims[4] = {1, 1, 1, 1};
            for (auto i : range0_(n)) {
                idims[4 - i - 1] = dimsI[n - i - 1];
                odims[4 - i - 1] = dimsO[n - i - 1];
            }
            CUDNN_ASSERT(cudnnSetTensor4dDescriptor(d->x, CUDNN_TENSOR_NCHW, cudnnDataType, idims[0], idims[1], idims[2], idims[3]));
            CUDNN_ASSERT(cudnnSetTensor4dDescriptor(d->y, CUDNN_TENSOR_NCHW, cudnnDataType, odims[0], odims[1], odims[2], odims[3]));
        } else {
            std::vector<int> strideI(n), strideO(n);
            size_t stride[]{1, 1};
            for (auto i : range0_(n).rev()) {
                strideI[i] = stride[0];
                strideO[i] = stride[1];
                stride[0] *= dimsI[i];
                stride[1] *= dimsO[i];
            }
            CUDNN_ASSERT(cudnnSetTensorNdDescriptor(d->x, cudnnDataType, n, dimsI.data(), strideI.data()));
            CUDNN_ASSERT(cudnnSetTensorNdDescriptor(d->y, cudnnDataType, n, dimsO.data(), strideO.data()));
        }

        // clang-format off
        auto reduceOp = reduceType == ReduceType::Mean ? CUDNN_REDUCE_TENSOR_AVG
                      : reduceType == ReduceType::Sum  ? CUDNN_REDUCE_TENSOR_ADD
                      : reduceType == ReduceType::Min  ? CUDNN_REDUCE_TENSOR_MIN
                      : reduceType == ReduceType::Max  ? CUDNN_REDUCE_TENSOR_MAX
                      : reduceType == ReduceType::L1   ? CUDNN_REDUCE_TENSOR_NORM1
                      : reduceType == ReduceType::L2   ? CUDNN_REDUCE_TENSOR_NORM2
                      : reduceType == ReduceType::Prod ? CUDNN_REDUCE_TENSOR_MUL
                      : UNREACHABLEX(cudnnReduceTensorOp_t, "");
        // clang-format on
        CUDNN_ASSERT(cudnnSetReduceTensorDescriptor(
            d->reduce, reduceOp, cudnnDataType,
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

        auto handler = res.fetchOrStore<CudnnContext>()->handle;
        size_t idxWorkspaceSize, workspaceSize;
        // get workspace
        CUDNN_ASSERT(cudnnGetReductionIndicesSize(handler, d->reduce, d->x, d->y, &idxWorkspaceSize));
        CUDNN_ASSERT(cudnnGetReductionWorkspaceSize(handler, d->reduce, d->x, d->y, &workspaceSize));
        idxWorkspaceSize = mem_manager::alignBytes(idxWorkspaceSize, 256);

        // nvcc at c++11 doesn't support real move capture
        auto routine = [d_ = std::move(d),
                        idxWorkspaceSize,
                        workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // fetch cudnn handle from resources
            auto const &d = *d_;
            // reduce
            float alpha = 1, beta = 0;
            void *idxWorkspace = workspace,
                 *dataWorkspace = reinterpret_cast<uint8_t *>(workspace) + idxWorkspaceSize;
            CUDNN_ASSERT(cudnnReduceTensor(
                res.fetchOrStore<CudnnContext>()->handle, d.reduce,
                idxWorkspace, idxWorkspaceSize,
                dataWorkspace, workspaceSize,
                &alpha, d.x, inputs[0],
                &beta, d.y, outputs[0]));
        };
        return RoutineWorkspace(std::move(routine), idxWorkspaceSize + workspaceSize);
    }

}// namespace refactor::kernel
