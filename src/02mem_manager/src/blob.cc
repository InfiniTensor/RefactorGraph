#include "mem_manager/blob.h"
#include <cstdlib>
#include <utility>

namespace refactor::mem_manager {

    Blob::Blob(size_t bytes) : ptr(std::malloc(bytes)) {}
    Blob::~Blob() { std::free(std::exchange(ptr, nullptr)); }

    std::pair<std::shared_ptr<Blob>, void *>
    Blob::share(size_t bytes) {
        auto blob = std::shared_ptr<Blob>(new Blob(bytes));
        return {blob, blob->ptr};
    }

}// namespace refactor::mem_manager
