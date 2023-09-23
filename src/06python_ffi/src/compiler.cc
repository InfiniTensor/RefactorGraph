#include "compiler.h"
#include "common/error_handler.h"
#include "functions.h"

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
    Compiler::setInput(size_t index, int dataType, DimVec shape) {
        ASSERT(index < _g.internal().topology.globalInputsCount(),
               fmt::format("Input {} not exist", index));

        auto dataType_ = *common::DataType::parse(dataType);
        Shape shape_(shape.size(), DimExpr(1));
        std::transform(shape.begin(), shape.end(), shape_.begin(),
                       [](auto const &d) {
                           return std::holds_alternative<int64_t>(d)
                                      ? DimExpr(std::get<int64_t>(d))
                                      : DimExpr(std::get<std::string>(d));
                       });
        _g.internal().edges[index].tensor = Tensor::share(dataType_, std::move(shape_), {});
    }

    std::unordered_set<std::string>
    Compiler::fillEdgeInfo() { return _g.fillEdgeInfo(); }

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

        auto ans = py::array(py::dtype(getFormat(tensor.dataType)), std::move(shape), nullptr);
        if (tensor.data) { std::memcpy(ans.mutable_data(), tensor.data->ptr, ans.nbytes()); }
        return ans;
    }

}// namespace refactor::python_ffi
