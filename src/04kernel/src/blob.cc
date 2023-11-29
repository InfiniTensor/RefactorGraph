#include "kernel/blob.hh"
#include <cstdlib>

namespace refactor::kernel {

    Blob::Blob(size_t bytes) : _ptr(std::malloc(bytes)) {}
    Blob::~Blob() { std::free(std::exchange(_ptr, nullptr)); }

    std::pair<Arc<Blob>, void *>
    Blob::share(size_t bytes) {
        auto blob = Arc<Blob>(new Blob(bytes));
        auto ptr = blob->_ptr;
        return {std::move(blob), ptr};
    }
    Blob::operator void const *() const noexcept { return _ptr; }

}// namespace refactor::kernel
