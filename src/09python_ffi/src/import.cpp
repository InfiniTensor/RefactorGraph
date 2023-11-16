#include "import.h"
#include <execution>
#include <fstream>

namespace refactor::python_ffi {
    using namespace frontend;

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
    makeOp(Name opType, AttributeMap attrs) {
        std::unordered_map<Name, Attribute> attrs_;
        for (auto &[name, value] : attrs) {
            attrs_.insert({std::move(name), {std::move(value)}});
        }
        return std::make_shared<OpBox>(Operator::build(fmt::format("onnx::{}", opType), std::move(attrs_)));
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
            builder.topology.insert(
                {std::move(node), {std::move(inputs), std::move(outputs)}});
        }
        builder.nodes.reserve(nodes.size());
        for (auto &[name, operator_] : nodes) {
            auto node = Node{std::move(*operator_), name};
            builder.nodes.insert({std::move(name), std::move(node)});
        }
        for (auto &[name, tensor] : edges) {
            auto edge = Edge{std::move(tensor), name};
            auto it = builder.edges.find(name);
            ASSERT(it != builder.edges.end(),
                   fmt::format("edge {} not connected", name));
            it->second.tensor = std::move(edge.tensor);
        }
        return std::make_shared<Compiler>(Graph(builder.build()));
    }

}// namespace refactor::python_ffi
