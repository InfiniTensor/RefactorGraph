#ifndef PYTHON_FFI_EXECUTOR_H
#define PYTHON_FFI_EXECUTOR_H

#include "computation/graph.h"
#include "functions.h"

namespace refactor::python_ffi {
    using SharedTensor = Arc<frontend::Tensor>;

    class Executor {
        computation::Graph _graph;
        runtime::Stream _stream;

    public:
        Executor(computation::Graph, runtime::Stream);
        void setInput(count_t, pybind11::array);
        auto getOutput(count_t) -> pybind11::array;
        void run();
        void bench(bool sync);
        void trace(std::string path, std::string format);
        void debugInfo() const noexcept;
    };

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_EXECUTOR_H
