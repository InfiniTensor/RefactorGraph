#include "kernel_list.h"
#include <cstdint>
#include <fmt/core.h>
#include <unordered_map>

namespace proj_namespace::kernel_list {
    void try_sub_project() {
        fmt::print("This is 'kernel_list' lib.\n");
    }

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
        std::unordered_map<
            KernelType,
            std::unordered_map<std::string, Kernel>,
            KernelTypeHash>
            inner;

    public:
        static KernelList global() {
            static KernelList g;
            return g;
        }

        bool insert(
            std::string device,
            std::string op_type,
            std::string name,
            void *code,
            std::function<bool(KernelParam const &)> predicate) {
            return inner
                .emplace(
                    KernelType{std::move(device), std::move(op_type)},
                    std::unordered_map<std::string, Kernel>{})
                .first->second// it->map
                .emplace(
                    std::move(name),
                    Kernel{code, std::move(predicate)})
                .second;
        }

        std::vector<std::pair<std::string, void *>> get(
            KernelType const &type,
            KernelParam const &param) const {

            auto const it = inner.find(type);
            if (it != inner.end()) {
                std::vector<std::pair<std::string, void *>> ans;
                for (auto const &[name, kernel] : it->second) {
                    if (kernel.predicate(param)) {
                        ans.push_back({name, kernel.code});
                    }
                }
                return ans;
            } else {
                return {};
            }
        }
    };

    std::vector<std::pair<std::string, void *>>
    get_kernels(KernelType const &type,
                KernelParam const &param) {
        return KernelList::global().get(type, param);
    }

}// namespace proj_namespace::kernel_list
