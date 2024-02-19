#ifndef NCCL_COMMUNICATOR_HH
#define NCCL_COMMUNICATOR_HH

#include "kernel/attributes/communication.h"
#include "runtime/resource.h"
#include <nccl.h>

#define checkNcclError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (ncclSuccess != err) {                                              \
            fprintf(stderr, "NCCL error in %s:%i : %s.\n", __FILE__, __LINE__, \
                    ncclGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }


namespace refactor::kernel::nccl {

    inline ncclRedOp_t getRedOp(kernel::AllReduceType t) {
        switch (t) {
            case kernel::AllReduceType::Sum:
                return ncclSum;
            case kernel::AllReduceType::Avg:
                return ncclAvg;
            case kernel::AllReduceType::Min:
                return ncclMin;
            case kernel::AllReduceType::Max:
                return ncclMax;
            case kernel::AllReduceType::Prod:
                return ncclProd;
            default:
                return ncclSum;
        }
    }

    inline ncclDataType_t getNcclDataType(DataType dataType) {
        switch (dataType) {
            case DataType::F32:
                return ncclFloat32;
            case DataType::U8:
                return ncclUint8;
            case DataType::I8:
                return ncclInt8;
            case DataType::I32:
                return ncclInt32;
            case DataType::I64:
                return ncclInt64;
            case DataType::FP16:
                return ncclFloat16;
            case DataType::F64:
                return ncclFloat64;
            case DataType::U32:
                return ncclUint32;
            case DataType::BF16:
                return ncclBfloat16;
            default:
                RUNTIME_ERROR("Datatype not supported by NCCL.");
        }
    }


    class NcclCommunicator final : public runtime::Resource {
    private:
        ncclComm_t comm;
        int const worldSize_, rank_;

    public:
        NcclCommunicator(int worldSize, int rank);
        ~NcclCommunicator();
        ncclComm_t get() { return comm; }
        int getWorldSize() { return worldSize_; }
        int getRank() { return rank_; }
        static size_t typeId() noexcept;
        static runtime::ResourceBox build(int worldSize, int rank) noexcept;

        size_t resourceTypeId() const noexcept final;
        std::string_view description() const noexcept final;
    };

}// namespace refactor::kernel::nccl

#endif
