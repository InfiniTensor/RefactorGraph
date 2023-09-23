#include "frontend/graph.h"
#include <pybind11/numpy.h>
#include <string>

namespace refactor::python_ffi {

    class Compiler {
        using CStr = char const *;
        using DimVec = std::vector<std::variant<std::string, int64_t>>;

        frontend::Graph _g;

    public:
        explicit Compiler(frontend::Graph);
        void substitute(CStr, int64_t);
        void setInput(size_t index, int dataType, DimVec shape);
        std::unordered_set<std::string> fillEdgeInfo();

        std::optional<pybind11::array> getTensor(const char *) const;
    };

}// namespace refactor::python_ffi
