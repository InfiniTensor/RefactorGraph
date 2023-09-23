#include "common/blob.h"
#include <cstdlib>
#include <utility>

namespace refactor::common {

    Blob::Blob(size_t byteSize) : ptr(std::malloc(byteSize)) {}
    Blob::~Blob() { std::free(std::exchange(ptr, nullptr)); }

}// namespace refactor::common
