#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include "hardware/functions.h"
#endif

namespace refactor::kernel {
    using K = ReduceCnnl;

    K::ReduceCnnl(
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
#ifndef USE_BANG
        return nullptr;
#endif

        auto const &x = inputs_[0].get();
        return x.dataType.isFloat()
                   ? std::make_unique<K>(x.dataType, reduceType_, std::move(axes_), x.shape)
                   : nullptr;
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing reduce operation using CNNL";
    }

#ifdef USE_BANG

    auto ReduceCnnl::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        // RAII for closure
        struct Descriptors {
            cnnlTensorDescriptor_t x, y;
            cnnlReduceDescriptor_t reduce;
            bool f32;

            explicit Descriptors(decltype(f32) f32_) : f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&x));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&y));
                CNNL_ASSERT(cnnlCreateReduceDescriptor(&reduce));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(x));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(y));
                CNNL_ASSERT(cnnlDestroyReduceDescriptor(reduce));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(dataType != DataType::F64);

        std::vector<int>
            dimsI(shape.begin(), shape.end()),
            dimsO(shape.begin(), shape.end());
        for (auto axis : axes) {
            dimsO[axis] = 1;
        }
        // setCnnlTensor(d->x, dataType, slice(dimsI.data(), dimsI.size()));
        // setCnnlTensor(d->y, dataType, slice(dimsO.data(), dimsO.size()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->x, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(dataType), dimsI.size(), dimsI.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->y, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(dataType), dimsO.size(), dimsO.data()));

        // clang-format off
        auto reduceOp = reduceType == ReduceType::Mean ? CNNL_REDUCE_AVG
                      : reduceType == ReduceType::Sum  ? CNNL_REDUCE_ADD
                      : reduceType == ReduceType::Min  ? CNNL_REDUCE_MIN
                      : reduceType == ReduceType::Max  ? CNNL_REDUCE_MAX
                      : reduceType == ReduceType::L1   ? CNNL_REDUCE_NORM1
                      : reduceType == ReduceType::L2   ? CNNL_REDUCE_NORM2
                      : reduceType == ReduceType::Prod ? CNNL_REDUCE_MUL
                      : UNREACHABLEX(cnnlReduceOp_t, "");
        // clang-format on
        CNNL_ASSERT(cnnlSetReduceDescriptor_v2(
            d->reduce, (int *) (axes.data()), axes.size(), reduceOp,
            cnnlDataTypeConvert(d->f32 ? DataType::F32 : DataType::F64),
            CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES, 0.0));

        auto handler = res.fetchOrStore<CnnlContext>()->handle;
        size_t idxWorkspaceSize = axes.size() * sizeof(int);
        // idxWorkspaceSize = hardware::alignBytes(idxWorkspaceSize, 256);
        size_t workspaceSize;
        // get workspace
        CNNL_ASSERT(cnnlGetReduceOpWorkspaceSize(handler, d->x, d->y, d->reduce, &workspaceSize));
        
        res.fetchOrStore<CnnlContext>();
        auto routine = [d = std::move(d),
                        idxWorkspaceSize,
                        workspaceSize](Resources &res,
                                       void *workspace,
                                       void const *const *inputs,
                                       void *const *outputs) {
            void *idxWorkspace = workspace,
                 *dataWorkspace = reinterpret_cast<uint8_t *>(workspace) + idxWorkspaceSize;
            // build alpha/beta for double
            auto a = d->f32 ? factor<fp32_t>(1) : factor<fp64_t>(1),
                 b = d->f32 ? factor<fp32_t>(0) : factor<fp64_t>(0);
            CNNL_ASSERT(cnnlReduce(
                res.fetchOrStore<CnnlContext>()->handle,
                d->reduce,
                dataWorkspace, workspaceSize,
                &a, d->x, inputs[0],
                idxWorkspaceSize, idxWorkspace,
                &b, d->y, outputs[0]));
        };
        return RoutineWorkspace(std::move(routine), idxWorkspaceSize + workspaceSize);
    }

#endif

}// namespace refactor::kernel
