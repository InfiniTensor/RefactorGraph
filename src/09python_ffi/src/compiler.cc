#include "compiler.h"
#include "common.h"
#include "kernel/allocators.h"
#include <execution>

namespace refactor::python_ffi {
    using namespace frontend;
    namespace py = pybind11;

    Compiler::Compiler(frontend::Graph g) : _g(std::move(g)) {}

    void
    Compiler::substitute(CStr name, int64_t value) {
        ASSERT(_g.substitute(name, value),
               fmt::format("Variable {} not exist", name));
    }

    void
    Compiler::setInput(size_t index, int dataType, DimVec dims) {
        ASSERT(index < _g.internal().topology.globalInputsCount(),
               fmt::format("Input {} not exist", index));

        auto dataType_ = *DataType::parse(dataType);
        _g.internal().edges[index].tensor = Tensor::share(dataType_, dimVec2Shape(dims), {});
    }

    std::unordered_set<std::string>
    Compiler::fillEdgeInfo(bool calculate) { return _g.fillEdgeInfo(calculate); }

    Arc<Executor>
    Compiler::compile(std::string target,
                      std::string allocator,
                      std::vector<std::string> passes) {
        _g.collectVariables();
        std::vector<std::string_view> unknownVariables;
        for (auto const &[_, v] : _g.variables()) {
            if (!v->value.has_value()) {
                unknownVariables.emplace_back(v->name);
            }
        }
        if (!unknownVariables.empty()) {
            std::string msg = "Unknown variables: [ ";
            for (auto const &v : unknownVariables) {
                msg += v;
                msg += ' ';
            }
            msg += ']';
            RUNTIME_ERROR(std::move(msg));
        }

        std::unordered_set<std::string> passes_;
        passes_.reserve(passes.size());
        for (auto &p : passes) { passes_.emplace(std::move(p)); }

        _g.fillEdgeInfo(passes_.erase("ce"));

        auto computation = _g.lower();
        if (passes_.find("lp") != passes_.end()) {
            computation.layoutPermute();
        }

        kernel::Target target_ = kernel::Target::Cpu;
        if (target == "cpu") {
            target_ = kernel::Target::Cpu;
        } else if (target == "cuda") {
            target_ = kernel::Target::NvidiaGpu;
        } else {
            UNREACHABLE();
        }

        auto kernel = computation.lower(target_);
        auto stream = kernel.lower(allocator == "flat"
                                       ? kernel::flatAllocate
                                       : kernel::reusableAllocate);

        return std::make_shared<Executor>(
            std::move(computation),
            std::move(stream));
    }

    std::vector<pybind11::array>
    Compiler::zeroInputs() const {
        std::vector<pybind11::array> ans;
        ans.reserve(_g.internal().topology.globalInputsCount());
        for (auto i : _g.internal().topology.globalInputs()) {
            auto const &tensor = *_g.internal().edges[i].tensor;
            ASSERT(!tensor.data, "Input tensor should not have data");

            std::vector<int64_t> shape(tensor.rank());
            std::transform(std::execution::unseq,
                           tensor.shape.begin(), tensor.shape.end(), shape.begin(),
                           [](auto const &d) { return d.value(); });
            ans.push_back(py::array(buildNumpyDType(tensor.dataType), std::move(shape)));
        }
        return ans;
    }

    std::optional<py::array>
    Compiler::getTensor(CStr name) const {
        auto const &edges = _g.internal().edges;
        auto it = std::find_if(edges.begin(), edges.end(),
                               [name](auto const &edge) { return edge.name == name; });
        if (it == edges.end()) { return std::nullopt; }
        auto const &tensor = *it->tensor;

        std::vector<int64_t> shape(tensor.shape.size());
        std::transform(std::execution::unseq,
                       tensor.shape.begin(), tensor.shape.end(), shape.begin(),
                       [](auto const &d) { return d.value(); });

        auto ans = py::array(buildNumpyDType(tensor.dataType), std::move(shape));
        if (tensor.data) { std::memcpy(ans.mutable_data(), tensor.data->get<void>(), ans.nbytes()); }
        return ans;
    }

}// namespace refactor::python_ffi
