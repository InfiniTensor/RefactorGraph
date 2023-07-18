#include "core.h"
#include <functional>
#include <string>

namespace proj_namespace::kernel_list {
    /// @brief 一个维度的信息。
    struct DimInfo {
        /// @brief 维度的长度。
        size_t size;
        /// @brief 维度的 layout 名称。
        std::string layout;
    };

    /// @brief kernel 的输入张量信息。
    struct KernelInput {
        /// @brief 张量的数据类型。
        core::DataType data_type;
        /// @brief 张量的形状。
        std::vector<DimInfo> shape;
    };

    /// @brief kernel 的属性张量信息。
    struct KernelAttribute {
        /// @brief 张量的数据类型。
        core::DataType data_type;
        /// @brief 张量的形状。
        std::vector<size_t> shape;
        /// @brief 张量的值。
        void *data;
    };

    /// @brief kernel 的参数。
    struct KernelParam {
        /// @brief kernel 的输入张量。
        std::vector<KernelInput> inputs;
        /// @brief kernel 的属性张量。
        std::unordered_map<std::string, KernelAttribute> attributes;
    };

    /// @brief kernel 信息。
    struct Kernel {
        /// @brief kernel 的名字。
        std::string name;
        /// @brief kernel 入口。
        void *code;
        /// @brief 测试 kernel 适用性的谓词。
        std::function<bool(KernelParam const &)> predicate;
    };

    /// @brief 获取 kernel 参数获取满足条件的 kernel。
    /// @param device 设备类型。
    /// @param op_type 算子类型。
    /// @param param kernel 参数。
    /// @return 符合要求的 kernel。
    std::vector<std::pair<std::string, void *>>
    get_kernels(
        std::string device,
        std::string op_type,
        KernelParam const &param);

    void try_sub_project();

}// namespace proj_namespace::kernel_list
