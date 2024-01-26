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
        void dispatch(Arc<hardware::Device>, std::string allocator);
        void setInput(count_t, pybind11::array);
        void setInputBlob(count_t, Arc<hardware::Device::Blob>);
        auto getOutput(count_t) const -> pybind11::array;
        auto getOutputBlob(count_t) const -> Arc<hardware::Device::Blob>;
        void run();
        void bench(bool sync);
        void trace(std::string path, std::string format);
        void debugInfo() const noexcept;
    };

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_EXECUTOR_H
