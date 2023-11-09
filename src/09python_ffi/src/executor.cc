#include "executor.h"

namespace refactor::python_ffi {

    Executor::Executor(computation::Graph graph, runtime::Stream stream)
        : _graph(std::move(graph)),
          _stream(std::move(stream)) {}

    void Executor::setInput(uint_lv1 i, pybind11::array data) {
        _stream.setInput(i, data.data(), data.nbytes());
    }

    auto Executor::getOutput(uint_lv1 i) -> pybind11::array {
        auto globalOutputs = _graph.internal().contiguous().topology.globalOutputs();
        ASSERT(i < globalOutputs.size(), "input index out of range");

        auto const &tensor = *_graph.internal().contiguous().edges[globalOutputs[i]].tensor;
        auto ans = pybind11::array(buildNumpyDType(tensor.dataType), std::move(tensor.shape), nullptr);
        _stream.getOutput(i, ans.mutable_data(), ans.nbytes());
        return ans;
    }

    auto Executor::prepare() -> std::vector<uint_lv1> {
        return _stream.prepare();
    }

    void Executor::run() {
        _stream.run();
    }

    void Executor::debugInfo() const noexcept {
        auto const &nodes = _graph.internal().contiguous().nodes;
        for (auto i : range0_(nodes.size())) {
            fmt::println("{}. {}", i, nodes[i].name);
        }
    }

}// namespace refactor::python_ffi
