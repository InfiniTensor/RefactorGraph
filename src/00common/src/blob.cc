#include "common/blob.h"
#include <cstdlib>
#include <utility>

namespace refactor::common {

    Blob::Blob(size_t byteSize)
        : ptr(std::malloc(byteSize)),
          dataType(DataType::F32),
          size(0),
          stamp(0) {}
    Blob::~Blob() { std::free(std::exchange(ptr, nullptr)); }

    size_t Blob::hash() const {
        size_t ret = reinterpret_cast<size_t>(ptr);
        ret += static_cast<size_t>(dataType.internal) << 1;
        ret += size << 2;
        ret += stamp << 3;
        return ret;
    }

}// namespace refactor::common
