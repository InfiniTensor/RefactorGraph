#include "executor.h"

namespace refactor::python_ffi {

    Executor::Executor(kernel::Graph graph, kernel::Allocator allocator)
        : _graph(std::move(graph)),
          _allocator(allocator),
          _stream(_graph.lower(_allocator)) {}

    void Executor::setInput(uint_lv1 i, pybind11::array data) {
        _stream.setInput(i, data.data(), data.nbytes());
    }

    std::vector<uint_lv1> Executor::prepare() {
        return _stream.prepare();
    }

    void Executor::run() {
        _stream.run();
    }

}// namespace refactor::python_ffi
