#include "compiler.h"
#include "common/error_handler.h"

namespace refactor::python_ffi {
    using namespace frontend;
    namespace py = pybind11;

    Compiler::Compiler(frontend::Graph g)
        : _g(std::move(g)) {}

    void
    Compiler::substitute(CStr name, int64_t value) {
        ASSERT(_g.substitute(name, value),
               fmt::format("Variable {} not exist", name));
    }

    void
    Compiler::setInput(size_t index, int dataType, DimVec dims) {
        ASSERT(index < _g.internal().topology.globalInputsCount(),
               fmt::format("Input {} not exist", index));

        auto dataType_ = *common::DataType::parse(dataType);
        _g.internal().edges[index].tensor = Tensor::share(dataType_, dimVec2Shape(dims), {});
    }

    std::unordered_set<std::string>
    Compiler::fillEdgeInfo() { return _g.fillEdgeInfo(); }

    Graph const &
    Compiler::graph() const { return _g; }

    std::optional<py::array>
    Compiler::getTensor(CStr name) const {
        auto const &edges = _g.internal().edges;
        auto it = std::find_if(edges.begin(), edges.end(),
                               [name](auto const &edge) { return edge.name == name; });
        if (it == edges.end()) { return std::nullopt; }
        auto const &tensor = *it->tensor;

        std::vector<int64_t> shape(tensor.shape.size());
        std::transform(tensor.shape.begin(), tensor.shape.end(), shape.begin(),
                       [](auto const &d) { return d.value(); });

        auto ans = py::array(buildNumpyDType(tensor.dataType), std::move(shape), nullptr);
        if (tensor.data) { std::memcpy(ans.mutable_data(), tensor.data->ptr, ans.nbytes()); }
        return ans;
    }

    std::shared_ptr<computation::Graph>
    Compiler::lower() const {
        return std::make_shared<computation::Graph>(_g.lower());
    }

}// namespace refactor::python_ffi
