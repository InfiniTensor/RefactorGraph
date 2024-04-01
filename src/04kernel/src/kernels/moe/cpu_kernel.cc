#include "cpu_kernel.hh"
#include <execution>
#include <list>

namespace refactor::kernel {

    AssignPosCpu::AssignPosCpu(AssignPosInfo info) noexcept
        : Kernel(), info(std::move(info)) {}

    auto AssignPosCpu::build(AssignPosInfo info) noexcept -> KernelBox {
        return std::make_unique<AssignPosCpu>(std::move(info));
    }
    auto AssignPosCpu::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto AssignPosCpu::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto AssignPosCpu::description() const noexcept -> std::string_view {
        return "Performing AssignPos operation on generic cpu";
    }

    auto AssignPosCpu::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;
        return [info = this->info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto gate = reinterpret_cast<int64_t const *>(inputs[0]);
             
            auto expert_cnt = reinterpret_cast<int64_t*>(outputs[0]);//T
            auto pos = reinterpret_cast<int64_t*>(outputs[1]);
            std::memset(expert_cnt, 0, info.expert_num);
            for (size_t i = 0; i < info.elementSize; i ++){
                ASSERT (gate[i] >= 0 && gate[i] < info.expert_num, "gate exceeds expert idx scope!");
                expert_cnt[gate[i]] ++;
            }
            std::vector<int64_t> expert_accumlate;
            expert_accumlate.assign(info.expert_num, 0);
            for (size_t i=0; i<expert_accumlate.size(); ++i){
                expert_accumlate[i] = (i==0) ? expert_cnt[i] : (expert_accumlate[i-1] + expert_cnt[i]);
            }

            for (size_t i=0; i< info.elementSize; ++i){
                pos[--expert_accumlate[gate[i]]] = i;
            }          
        };
    }


    ReorderCpu::ReorderCpu(ReorderInfo info) noexcept
        : Kernel(), info(std::move(info)) {}

    auto ReorderCpu::build(ReorderInfo info) noexcept -> KernelBox {
        return std::make_unique<ReorderCpu>(std::move(info));
    }
    auto ReorderCpu::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto ReorderCpu::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto ReorderCpu::description() const noexcept -> std::string_view {
        return "Performing scatter operation on generic cpu";
    }

    auto ReorderCpu::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;
        return [info = this->info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto input = reinterpret_cast<float const *>(inputs[0]);
            auto pos = reinterpret_cast<int64_t const *>(inputs[1]);
            auto dstVal = reinterpret_cast<float*>(outputs[0]);//T
                             
            for(size_t i = 0; i<info.blockNum; i++){
                if (info.scatter)
                    std::copy_n(input + (pos[i]/info.top) * info.blockSize, info.blockSize, dstVal + i*info.blockSize);
                else 
                    std::copy_n(input + i*info.blockSize, info.blockSize, dstVal + pos[i] * info.blockSize);
            }            
        };
    }
}// namespace refactor::kernel
