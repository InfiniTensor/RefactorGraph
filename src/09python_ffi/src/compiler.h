#ifndef PYTHON_FFI_COMPILER_H
#define PYTHON_FFI_COMPILER_H

#include "executor.h"
#include "frontend/graph.h"
#include "functions.h"

namespace refactor::python_ffi {

    class Compiler {
        using CStr = char const *;

        frontend::Graph _g;

    public:
        explicit Compiler(frontend::Graph);
        void substitute(CStr, int64_t);
        void setInput(size_t index, int dataType, DimVec dims);
        std::unordered_set<std::string> fillEdgeInfo(bool calculate);
        Arc<Executor> compileOn(
            Arc<hardware::Device> device,
            std::string allocator,
            std ::vector<std::string> passes);
        Arc<Executor> compile(
            std::string target,
            std::string allocator,
            std ::vector<std::string> passes);

        std::vector<pybind11::array> zeroInputs() const;
        std::optional<pybind11::array> getTensor(CStr) const;

        void serialize(std::string path);
    };

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_COMPILER_H
