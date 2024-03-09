#include "cpu_kernel.hh"
#include <execution>
#include <list>

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
                       
            
            size_t M = info.getElementSize() / info.getAxisElementSize();
            size_t N = info.getAxisElementSize();
            auto inStride1 = info.getInStridePreAxis();
            auto inStride2 = info.getInStride();
            auto outStride1 = info.getOutStridePreAxis();
            auto outStride2 = inStride2;

            for(size_t m = 0; m < M; m ++){
                using PairType = std::pair<float, uint8_t>;
                std::list<PairType> list;
                for(size_t n = 0; n < N; n++){                    
                    auto srcIdx = m /inStride2 * inStride1 + m % inStride2 + n * inStride2;                    
                    list.push_back({src[srcIdx],n});                    
                }
                list.sort([](const PairType &a, const PairType &b)->bool{return a.first > b.first;});
                
                size_t offset = m /inStride2 * outStride1 + m % inStride2;
                std::for_each_n(list.begin(), (uint32_t)info.topk,
                            [&](auto &elem) {                                
                                dstVal[offset] = elem.first;
                                dstIndex[offset] = elem.second;
                                offset += outStride2;
                            });   
            }
        };
    }
}// namespace refactor::kernel
