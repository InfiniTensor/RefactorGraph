#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#endif


namespace refactor::kernel {
    using K = CastCnnl;
    using DT = DataType;

    K::CastCnnl(decltype(from) from_,
                decltype(to) to_,
                decltype(shape) shape_) noexcept
        : from(from_), to(to_), shape(shape_) {}

    auto K::build(Tensor const &from, Tensor const &to) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif

        return std::make_unique<K>(from.dataType, to.dataType,
                                   std::vector<int>(from.shape.begin(), from.shape.end()));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing cast operation using CNNL";
    }

#ifdef USE_BANG

    static cnnlCastDataType_t castType(DataType from, DataType to);

    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        struct Descriptors {
            cnnlTensorDescriptor_t inDesc, outDesc;
            cnnlCastDataType_t cast;

            Descriptors() : inDesc(nullptr), outDesc(nullptr) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&inDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&outDesc));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(inDesc));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(outDesc));
            }
        };
        auto d = std::make_shared<Descriptors>();
        d->cast = castType(from, to);
        setCnnlTensor(d->inDesc, from, slice(shape.data(), shape.size()));
        setCnnlTensor(d->outDesc, to, slice(shape.data(), shape.size()));

        res.fetchOrStore<CnnlContext>();
        return [d = std::move(d)](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            CNNL_ASSERT(cnnlCastDataType(res.fetchOrStore<CnnlContext>()->handle,
                                         d->inDesc, inputs[0], d->cast, d->outDesc, outputs[0]));
        };
    }

    static cnnlCastDataType_t castType(DataType from, DataType to) {
        switch (from) {
            case DT::F32:
                switch (to) {
                    case DT::F64:
                        return CNNL_CAST_FLOAT_TO_DOUBLE;
                    case DT::FP16:
                        return CNNL_CAST_FLOAT_TO_HALF;
                    case DT::I64:
                        return CNNL_CAST_FLOAT_TO_INT64;
                    case DT::I32:
                        return CNNL_CAST_FLOAT_TO_INT32;
                    case DT::I16:
                        return CNNL_CAST_FLOAT_TO_INT16;
                    case DT::I8:
                        return CNNL_CAST_FLOAT_TO_INT8;
                    case DT::U8:
                        return CNNL_CAST_FLOAT_TO_UINT8;
                    // case DT::BF16:
                    //     return CNNL_CAST_FLOAT_TO_BFLOAT16;
                    case DT::Bool:
                        return CNNL_CAST_FLOAT_TO_BOOL;
                    default:
                        UNREACHABLE();
                }
            case DT::FP16:
                switch (to) {
                    case DT::F32:
                        return CNNL_CAST_HALF_TO_FLOAT;
                    case DT::I64:
                        return CNNL_CAST_HALF_TO_INT64;
                    case DT::I32:
                        return CNNL_CAST_HALF_TO_INT32;
                    case DT::I16:
                        return CNNL_CAST_HALF_TO_INT16;
                    case DT::I8:
                        return CNNL_CAST_HALF_TO_INT8;
                    case DT::U8:
                        return CNNL_CAST_HALF_TO_UINT8;
                    case DT::Bool:
                        return CNNL_CAST_HALF_TO_BOOL;
                    default:
                        UNREACHABLE();
                }
            case DT::I32:
                switch (to) {
                    case DT::F32:
                        return CNNL_CAST_INT32_TO_FLOAT;
                    case DT::FP16:
                        return CNNL_CAST_INT32_TO_HALF;
                    case DT::I64:
                        return CNNL_CAST_INT32_TO_INT64;
                    case DT::I16:
                        return CNNL_CAST_INT32_TO_INT16;
                    case DT::I8:
                        return CNNL_CAST_INT32_TO_INT8;
                    case DT::Bool:
                        return CNNL_CAST_INT32_TO_BOOL;
                    default:
                        UNREACHABLE();
                }
            case DT::I16:
                switch (to) {
                    case DT::F32:
                        return CNNL_CAST_INT16_TO_FLOAT;
                    case DT::FP16:
                        return CNNL_CAST_INT16_TO_HALF;
                    case DT::I32:
                        return CNNL_CAST_INT16_TO_INT32;
                    // case DT::I8:
                    //     return CNNL_CAST_INT16_TO_INT8;
                    default:
                        UNREACHABLE();
                }
            case DT::I8:
                switch (to) {
                    case DT::F32:
                        return CNNL_CAST_INT8_TO_FLOAT;
                    case DT::FP16:
                        return CNNL_CAST_INT8_TO_HALF;
                    case DT::I32:
                        return CNNL_CAST_INT8_TO_INT32;
                    case DT::I16:
                        return CNNL_CAST_INT8_TO_INT16;
                    default:
                        UNREACHABLE();
                }
            case DT::U8:
                switch (to) {
                    case DT::F32:
                        return CNNL_CAST_UINT8_TO_FLOAT;
                    case DT::FP16:
                        return CNNL_CAST_UINT8_TO_HALF;
                    case DT::I64:
                        return CNNL_CAST_UINT8_TO_INT64;
                    case DT::I32:
                        return CNNL_CAST_UINT8_TO_INT32;
                    default:
                        UNREACHABLE();
                }
            case DT::Bool:
                switch (to) {
                    case DT::F32:
                        return CNNL_CAST_BOOL_TO_FLOAT;
                    case DT::FP16:
                        return CNNL_CAST_BOOL_TO_HALF;
                    case DT::I32:
                        return CNNL_CAST_BOOL_TO_INT32;
                    default:
                        UNREACHABLE();
                }
            case DT::I64:
                switch (to) {
                    case DT::F32:
                        return CNNL_CAST_INT64_TO_FLOAT;
                    case DT::FP16:
                        return CNNL_CAST_INT64_TO_HALF;
                    case DT::I32:
                        return CNNL_CAST_INT64_TO_INT32;
                    case DT::U32:
                        return CNNL_CAST_INT64_TO_UINT32;
                    default:
                        UNREACHABLE();
                }
            case DT::U32:
                switch (to) {
                    case DT::I64:
                        return CNNL_CAST_UINT32_TO_INT64;
                    case DT::U64:
                        return CNNL_CAST_UINT32_TO_UINT64;
                    default:
                        UNREACHABLE();
                }
            case DT::F64:
                switch (to) {
                    case DT::F32:
                        return CNNL_CAST_DOUBLE_TO_FLOAT;
                    default:
                        UNREACHABLE();
                }
            case DT::BF16:
                switch (to) {
                    // case DT::F32:
                    //     return CNNL_CAST_BF16_TO_FLOAT;
                    default:
                        UNREACHABLE();
                }
            default:
                UNREACHABLE();
        }
    }

#endif

}// namespace refactor::kernel
