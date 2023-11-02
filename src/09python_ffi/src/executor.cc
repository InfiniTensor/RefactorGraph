#include "executor.h"

namespace refactor::python_ffi {

    Executor::Executor(kernel::Graph graph, kernel::Allocator allocator)
        : _graph(std::move(graph)),
          _allocator(allocator),
          _stream(_graph.lower(_allocator)) {}

    void Executor::setInput(uint_lv1 i, SharedTensor tensor) {
        _stream.setInput(i, tensor->data->operator const void *(), tensor->bytesSize());
    }

    std::vector<uint_lv1> Executor::prepare() {
        return _stream.prepare();
    }

}// namespace refactor::python_ffi
