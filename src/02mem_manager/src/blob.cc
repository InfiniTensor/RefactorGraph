#include "mem_manager/blob.hh"
#include <cstdlib>
#include <utility>

namespace refactor::mem_manager {

    Blob::Blob(size_t bytes) : _ptr(std::malloc(bytes)) {}
    Blob::~Blob() { std::free(std::exchange(_ptr, nullptr)); }

    std::pair<std::shared_ptr<Blob>, void *>
    Blob::share(size_t bytes) {
        auto blob = std::shared_ptr<Blob>(new Blob(bytes));
        auto ptr = blob->_ptr;
        return {std::move(blob), ptr};
    }

}// namespace refactor::mem_manager
