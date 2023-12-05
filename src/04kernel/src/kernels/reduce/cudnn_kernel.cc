#include "cudnn_kernel.hh"

#ifdef USE_CUDA
#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "hardware/functions.h"
#endif

namespace refactor::kernel {
    using K = ReduceCudnn;

    K::ReduceCudnn(
        decltype(dataType) dataType_,
        decltype(reduceType) reduceType_,
        decltype(axes) axes_,
        decltype(shape) shape_) noexcept
        : Kernel(),
          dataType(dataType_),
          reduceType(reduceType_),
          axes(std::move(axes_)),
          shape(std::move(shape_)) {}

    auto K::build(decltype(axes) axes_, ReduceType reduceType_, TensorRefs inputs_) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        auto const &x = inputs_[0].get();
        return x.dataType.isCpuNumberic()
                   ? std::make_unique<K>(x.dataType, reduceType_, std::move(axes_), x.shape)
                   : nullptr;
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing reduce operation using CUDNN";
    }

#ifdef USE_CUDA

    auto ReduceCudnn::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cudnn;
        using namespace runtime;

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
            ~Descriptors() noexcept(false) {
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

        std::vector<int>
            dimsI(shape.begin(), shape.end()),
            dimsO(shape.begin(), shape.end());
        for (auto axis : axes) {
            dimsO[axis] = 1;
        }
        setCudnnTensor(d->x, dataType, slice(dimsI.data(), dimsI.size()));
        setCudnnTensor(d->y, dataType, slice(dimsO.data(), dimsO.size()));

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
            d->reduce, reduceOp, cudnnDataTypeConvert(dataType),
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

        auto handler = res.fetchOrStore<CudnnContext>()->handle;
        size_t idxWorkspaceSize, workspaceSize;
        // get workspace
        CUDNN_ASSERT(cudnnGetReductionIndicesSize(handler, d->reduce, d->x, d->y, &idxWorkspaceSize));
        CUDNN_ASSERT(cudnnGetReductionWorkspaceSize(handler, d->reduce, d->x, d->y, &workspaceSize));
        idxWorkspaceSize = hardware::alignBytes(idxWorkspaceSize, 256);

        // nvcc at c++11 doesn't support real move capture
        auto routine = [d = std::move(d),
                        idxWorkspaceSize,
                        workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            void *idxWorkspace = workspace,
                 *dataWorkspace = reinterpret_cast<uint8_t *>(workspace) + idxWorkspaceSize;
            float alpha = 1, beta = 0;
            CUDNN_ASSERT(cudnnReduceTensor(
                res.fetchOrStore<CudnnContext>()->handle,
                d->reduce,
                idxWorkspace, idxWorkspaceSize,
                dataWorkspace, workspaceSize,
                &alpha, d->x, inputs[0],
                &beta, d->y, outputs[0]));
        };
        return RoutineWorkspace(std::move(routine), idxWorkspaceSize + workspaceSize);
    }

#endif

}// namespace refactor::kernel
