#include "cpu_kernel.hh"
#include <execution>
#include <vector>

namespace refactor::kernel {
    using K = TopKCpu;

    K::TopKCpu(TopKInfo info) noexcept
        : Kernel(), info(std::move(info)) {}

    auto K::build(TopKInfo info) noexcept -> KernelBox {
        return std::make_unique<K>(std::move(info));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing topk operation on generic cpu";
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;
        return [info = this->info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto src = reinterpret_cast<float const *>(inputs[0]);
            
            auto dstVal = reinterpret_cast<float*>(outputs[0]);//T
            auto dstIndex = reinterpret_cast<uint32_t*>(outputs[1]);
                       
            
            size_t M = info.size.except_axis; 
            size_t N = info.size.axis;

            for(size_t m = 0; m < M; m ++){
                using PairType = std::pair<float, uint32_t>;
                std::vector<PairType> list;
                for(size_t n = 0; n < N; n++){                    
                    auto srcIdx = m /info.stride.axis * info.stride.in_pre + m % info.stride.axis + n * info.stride.axis;                    
                    list.push_back({src[srcIdx],n});                    
                }
                //list.sort([](const PairType &a, const PairType &b)->bool{return a.first > b.first;});
                std::partial_sort(list.begin(), \
                    list.begin() + info.topk, \
                    list.end(), \
                    [](const PairType &a, const PairType &b)->bool{return a.first > b.first;});
                
                size_t offset = m /info.stride.axis * info.stride.out_pre + m % info.stride.axis;
                std::for_each_n(list.begin(), (uint32_t)info.topk,
                            [&](auto &elem) {                                
                                dstVal[offset] = elem.first;
                                dstIndex[offset] = elem.second;
                                offset += info.stride.axis;
                            });   
            }
        };
    }
}// namespace refactor::kernel
