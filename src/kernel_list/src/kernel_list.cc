#include "kernel_list.h"
#include <cstdint>
#include <fmt/core.h>
#include <unordered_map>

namespace proj_namespace::kernel_list {
    void try_sub_project() {
        fmt::print("This is 'kernel_list' lib.\n");
    }

    struct KernelType {
        std::string device;
        std::string op_type;

        bool operator==(KernelType const &others) const {
            return device == others.device && op_type == others.op_type;
        }
    };

    struct KernelTypeHash {
        size_t operator()(KernelType const &kt) const {
            size_t h[]{
                std::hash<std::string>{}(kt.device),
                std::hash<std::string>{}(kt.op_type),
            };
            return h[0] ^ (h[1] << 1);
        }
    };

    class KernelList {
        std::unordered_multimap<KernelType, Kernel, KernelTypeHash> inner;

    public:
        static KernelList global() {
            static KernelList g;
            return g;
        }

        std::vector<std::pair<std::string, void *>>
        get(std::string device,
            std::string op_type,
            KernelParam const &param) const {
            std::vector<std::pair<std::string, void *>> ans;

            auto [begin, end] = inner.equal_range({std::move(device), std::move(op_type)});
            for (auto it = begin; it != end; ++it) {
                auto const &[name, code, predicate] = it->second;
                if (predicate(param)) {
                    ans.push_back({name, code});
                }
            }

            return ans;
        }
    };

    std::vector<std::pair<std::string, void *>>
    get_kernels(std::string device,
                std::string op_type,
                KernelParam const &param) {
        return KernelList::global().get(std::move(device), std::move(op_type), param);
    }

}// namespace proj_namespace::kernel_list
