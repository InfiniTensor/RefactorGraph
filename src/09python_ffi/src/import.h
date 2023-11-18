#ifndef PYTHON_FFI_IMPORT_H
#define PYTHON_FFI_IMPORT_H

#include "compiler.h"

namespace refactor::python_ffi {
    using SharedTensor = Arc<frontend::Tensor>;
    using SharedOp = Arc<frontend::OpBox>;
    using Name = std::string;
    using NameVec = std::vector<Name>;
    using AttributeMap = std::unordered_map<Name, decltype(frontend::Attribute::value)>;

    SharedTensor makeTensor(int dataType, DimVec dims);
    SharedTensor makeTensorWithData(pybind11::array);
    SharedTensor makeTensorWithExternalData(
        int dataType,
        std::vector<int64_t> shape,
        std::string file,
        int64_t offset);
    SharedOp makeOp(AttributeMap, Name opType, AttributeMap);
    Arc<Compiler> makeCompiler(
        std::unordered_map<Name, std::pair<NameVec, NameVec>> topology,
        std::unordered_map<Name, SharedOp> nodes,
        std::unordered_map<Name, SharedTensor> edges,
        NameVec inputs,
        NameVec outputs);

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_IMPORT_H
