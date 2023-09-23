#ifndef PYTHON_FFI_COMPILER_H
#define PYTHON_FFI_COMPILER_H

#include "frontend/graph.h"
#include "functions.h"
#include <pybind11/numpy.h>
#include <string>

namespace refactor::python_ffi {

    class Compiler {
        using CStr = char const *;

        frontend::Graph _g;

    public:
        explicit Compiler(frontend::Graph);
        void substitute(CStr, int64_t);
        void setInput(size_t index, int dataType, DimVec dims);
        std::unordered_set<std::string> fillEdgeInfo();

        frontend::Graph const &graph() const;
        std::optional<pybind11::array> getTensor(CStr) const;
        std::shared_ptr<computation::Graph> lower() const;
    };

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_COMPILER_H
