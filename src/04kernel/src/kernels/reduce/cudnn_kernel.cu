#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "common.h"
#include "cudnn_kernel.hh"
#include "runtime/mem_manager.hh"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;

    Routine ReduceCudnn::lower(Resources &res) const noexcept {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t inDesc;
            cudnnTensorDescriptor_t outDesc;

            cudnnReduceTensorDescriptor_t reduceDesc;
            size_t workspaceSize;
            size_t idxWorkspaceSize;

            Descriptors() : workspaceSize(0), idxWorkspaceSize(0) {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&inDesc));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&outDesc));
                CUDNN_ASSERT(cudnnCreateReduceTensorDescriptor(&reduceDesc));
            }
            ~Descriptors() {
                // Destories in CUDA does not require sync. But cuDNN does not state
                // whether sync is required before destories.
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(inDesc));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
                CUDNN_ASSERT(cudnnDestroyReduceTensorDescriptor(reduceDesc));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();
        auto handler = res.fetchOrStore<CudnnContext>()->handle;

        // Each dimension of the output tensor C must match the corresponding
        // dimension of the input tensor A or must be equal to 1. The dimensions
        // equal to 1 indicate the dimensions of A to be reduced.
        int nInDims = shape.size();
        std::vector<int> inDimArray, outDimArray, inStrideArray, outStrideArray;
        size_t stride = 1;
        for (int i = nInDims - 1; i >= 0; --i) {
            inDimArray.insert(inDimArray.begin(), shape[i]);
            inStrideArray.insert(inStrideArray.begin(), stride);
            stride *= shape[i];
        }
        std::unordered_set axesSet(axes.begin(), axes.end());
        for (size_t i = 0; i < shape.size(); ++i) {
            if (axesSet.find(i) == axesSet.end()) {
                outDimArray.push_back(shape[i]);
            } else {
                outDimArray.push_back(1);
            }
        }
        size_t nOutDims = outDimArray.size();
        stride = 1;
        for (int i = nOutDims - 1; i >= 0; --i) {
            outStrideArray.insert(outStrideArray.begin(), stride);
            stride *= outDimArray[i];
        }

        // cudnnSetTensorNdDescriptor is used when nDim>3, otherwise,it is
        // recomended to use cudnnSetTensor4dDescriptor and set the unused
        // dimension size to 1.
        // get inputs outputs
        auto cudnnDataType = cudnnDataTypeConvert(dataType);
        if (nInDims > 3) {
            CUDNN_ASSERT(cudnnSetTensorNdDescriptor(
                d->inDesc, cudnnDataType, nInDims, inDimArray.data(), inStrideArray.data()));
            CUDNN_ASSERT(cudnnSetTensorNdDescriptor(
                d->outDesc, cudnnDataType, nOutDims, outDimArray.data(), outStrideArray.data()));
        } else {
            int idims[4] = {1, 1, 1, 1}, odims[4] = {1, 1, 1, 1};
            for (int i = 0; i < nInDims; ++i) {
                idims[4 - i - 1] = inDimArray[nInDims - i - 1];
            }
            for (int i = 0; i < nOutDims; ++i) {
                odims[4 - i - 1] = outDimArray[nOutDims - i - 1];
            }

            CUDNN_ASSERT(cudnnSetTensor4dDescriptor(
                d->inDesc, CUDNN_TENSOR_NCHW, cudnnDataType, idims[0], idims[1],
                idims[2], idims[3]));
            CUDNN_ASSERT(cudnnSetTensor4dDescriptor(
                d->outDesc, CUDNN_TENSOR_NCHW, cudnnDataType, odims[0],
                odims[1], odims[2], odims[3]));
        }

        // get reduce descriptor
        cudnnReduceTensorOp_t reduceOp = CUDNN_REDUCE_TENSOR_ADD;
        switch (reduceType) {
            case ReduceType::Mean:
                reduceOp = CUDNN_REDUCE_TENSOR_AVG;
                break;
            case ReduceType::Min:
                reduceOp = CUDNN_REDUCE_TENSOR_MIN;
                break;
            case ReduceType::Max:
                reduceOp = CUDNN_REDUCE_TENSOR_MAX;
                break;
            case ReduceType::L1:
                reduceOp = CUDNN_REDUCE_TENSOR_NORM1;
                break;
            case ReduceType::L2:
                reduceOp = CUDNN_REDUCE_TENSOR_NORM2;
                break;
            case ReduceType::Sum:
                reduceOp = CUDNN_REDUCE_TENSOR_ADD;
                break;
            case ReduceType::Prod:
                reduceOp = CUDNN_REDUCE_TENSOR_MUL;
                break;
            default:
                UNREACHABLE();
        };
        CUDNN_ASSERT(cudnnSetReduceTensorDescriptor(
            d->reduceDesc, reduceOp, cudnnDataType,
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
            CUDNN_32BIT_INDICES));

        // get workspace
        CUDNN_ASSERT(
            cudnnGetReductionWorkspaceSize(handler, d->reduceDesc,
                                           d->inDesc, d->outDesc, &d->workspaceSize));

        // get index workspace
        CUDNN_ASSERT(
            cudnnGetReductionIndicesSize(handler, d->reduceDesc,
                                         d->inDesc, d->outDesc, &d->idxWorkspaceSize));


        // nvcc at c++11 doesn't support real move capture
        return [d_ = std::move(d)](Resources &res, void const **inputs, void **outputs) {
            using mem_manager::ForeignBlob;
            // fetch cudnn handle from resources
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
            auto const &d = *d_;
            auto wsData = ForeignBlob::share(res.fetch<runtime::MemManager>()->manager, d.workspaceSize);
            auto idxWsData = ForeignBlob::share(res.fetch<runtime::MemManager>()->manager, d.idxWorkspaceSize);

            // name inputs and outputs
            auto inData = inputs[0];
            auto outData = outputs[0];
            // reduce
            float alpha = 1.f, beta = 0.f;
            CUDNN_ASSERT(cudnnReduceTensor(handle, d.reduceDesc,
                                           *idxWsData, d.idxWorkspaceSize, *wsData,
                                           d.workspaceSize, &alpha, d.inDesc, inData,
                                           &beta, d.outDesc, outData));
        };
    }

}// namespace refactor::kernel
