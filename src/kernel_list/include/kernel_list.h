#include "core.h"
#include <functional>
#include <string>

namespace proj_namespace::kernel_list {

    struct DimInfo {
        size_t size;
        std::string layout;
    };

    struct KernelInput {
        core::DataType data_type;
        std::vector<DimInfo> shape;
    };

    struct KernelAttribute {
        core::DataType data_type;
        std::vector<size_t> shape;
        void *data;
    };

    struct KernelParam {
        std::vector<KernelInput> inputs;
        std::unordered_map<std::string, KernelAttribute> attributes;
    };

    struct Kernel {
        std::string name;
        void *code;
        std::function<bool(KernelParam const &)> predicate;
    };

    void try_sub_project();

}// namespace proj_namespace::kernel_list
