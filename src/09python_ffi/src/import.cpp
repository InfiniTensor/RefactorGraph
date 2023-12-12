#include "import.h"
#include "hardware/device_manager.h"
#include <execution>
#include <fstream>

namespace refactor::python_ffi {
    using namespace frontend;
    using namespace hardware;

    SharedDevice
    findDevice(std::string type, int card) {
        std::transform(std::execution::unseq,
                       type.begin(), type.end(),
                       type.begin(),
                       ::tolower);
        // clang-format off
        auto type_ = type == "cpu"    ? Device::Type::Cpu
                   : type == "nvidia" ? Device::Type::Nvidia
                   : UNREACHABLEX(Device::Type, "Unknown device type: \"{}\"", type);
        // clang-format on
        return device::init(type_, card, "");
    }

    SharedTensor
    makeTensor(int dataType, DimVec dims) {
        return Tensor::share(*DataType::parse(dataType), dimVec2Shape(dims), {});
    }

    SharedTensor
    makeTensorWithData(pybind11::array data) {
        Shape shape(data.ndim(), DimExpr(1));
        std::transform(std::execution::unseq,
                       data.shape(), data.shape() + data.ndim(), shape.begin(),
                       [](auto const &d) { return DimExpr(d); });
        auto ans = Tensor::share(parseNumpyDType(data.dtype()), std::move(shape), {});
        std::memcpy(ans->malloc(), data.data(), data.nbytes());
        return ans;
    }

    SharedTensor makeTensorWithExternalData(
        int dataType,
        std::vector<int64_t> shape,
        std::string file,
        int64_t offset) {
        Shape shape_(shape.size(), DimExpr(1));
        std::transform(std::execution::unseq,
                       shape.begin(), shape.end(), shape_.begin(),
                       [](auto d) { return DimExpr(d); });
        auto ans = Tensor::share(*DataType::parse(dataType), std::move(shape_), {});
        std::ifstream stream(file, std::ios::binary);
        ASSERT(stream.is_open(), "No such file: \"{}\"", file);
        stream.seekg(offset);
        stream.read(static_cast<char *>(ans->malloc()), ans->bytesSize());
        return ans;
    }

    SharedOp
    makeOp(AttributeMap ctx, Name opType, AttributeMap attrs) {
        std::unordered_map<Name, Attribute> attrs_;
        for (auto &[name, value] : attrs) {
            attrs_.insert({std::move(name), {std::move(value)}});
        }
        std::unordered_map<Name, Attribute> ctx_;
        for (auto &[name, value] : ctx) {
            ctx_.insert({std::move(name), {std::move(value)}});
        }
        return std::make_shared<OpBox>(Operator::build(
            ctx_, fmt::format("onnx::{}", opType), std::move(attrs_)));
    }

    Arc<Compiler>
    makeCompiler(
        std::unordered_map<Name, std::pair<NameVec, NameVec>> topology,
        std::unordered_map<Name, SharedOp> nodes,
        std::unordered_map<Name, SharedTensor> edges,
        NameVec inputs_,
        NameVec outputs_) {
        auto builder = graph_topo::Builder<Name, Node, Name, Edge>{
            {},
            std::move(inputs_),
            std::move(outputs_),
        };
        for (auto &[node, rels] : topology) {
            auto &[inputs, outputs] = rels;
            for (auto const &input : inputs) {
                builder.edges.insert({input, {nullptr, input}});
            }
            for (auto const &output : outputs) {
                builder.edges.insert({output, {nullptr, output}});
            }
            builder.topology.insert({
                std::move(node),
                {std::move(inputs), std::move(outputs)},
            });
        }
        builder.nodes.reserve(nodes.size());
        for (auto &[name, operator_] : nodes) {
            builder.nodes.insert({std::move(name), Node{std::move(*operator_), name}});
        }
        for (auto &[name, tensor] : edges) {
            if (auto it = builder.edges.find(name); it != builder.edges.end()) {
                it->second.tensor = std::move(tensor);
            } else {
                fmt::println("\x1b[93mWARNING: edge \"{}\" not connected\x1b[0m", name);
            }
        }
        return std::make_shared<Compiler>(Graph(builder.build()));
    }

}// namespace refactor::python_ffi
