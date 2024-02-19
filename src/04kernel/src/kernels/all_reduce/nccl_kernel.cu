#include "nccl_kernel.hh"
#include "../../utilities/cuda/nccl_communicator.hh"
#include <nccl.h>
namespace refactor::kernel {
    using K = AllReduceNccl;
    using DT = DataType;
    using namespace nccl;

    auto K::lower(Resources &res) const noexcept -> RoutineWorkspace{
        return [count = size,
                redOp = getRedOp(opType),
                ncclDataType = getNcclDataType(dataType)](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto communicator = res.fetch<NcclCommunicator>();
            auto input = inputs[0];
            auto output = outputs[0];
            checkNcclError(ncclAllReduce(input, output, count, ncclDataType,
                                         redOp, communicator->get(), 0));// TODO: use default stream for now
        };
    }
}// namespace refactor::kernel
