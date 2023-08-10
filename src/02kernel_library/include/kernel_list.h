#include "data_type.h"
#include <functional>
#include <string>

namespace refactor::kernel_list {
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
        common::DataType dataType;
        /// @brief 张量的形状。
        std::vector<DimInfo> shape;
    };

    /// @brief kernel 的属性张量信息。
    struct KernelAttribute {
        /// @brief 张量的数据类型。
        common::DataType dataType;
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

    /// @brief 用于索引的 kernel 类型。
    struct KernelType {
        /// @brief kernel 工作的设备。
        std::string device;
        /// @brief kernel 语义对应的算子。
        std::string opType;

        bool operator==(KernelType const &others) const {
            return device == others.device && opType == others.opType;
        }
    };

    /// @brief kernel 信息。
    struct Kernel {
        /// @brief kernel 入口。
        void *code;
        /// @brief 测试 kernel 适用性的谓词。
        std::function<bool(KernelParam const &)> predicate;
    };

    /// @brief 获取 kernel 参数获取满足条件的 kernel。
    /// @param device 设备类型。
    /// @param opType 算子类型。
    /// @param param kernel 参数。
    /// @return 符合要求的 kernel。
    std::vector<std::pair<std::string, void *>>
    getKernels(
        KernelType const &type,
        KernelParam const &param);

    void trySubProject();

}// namespace refactor::kernel_list
