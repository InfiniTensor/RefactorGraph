#include "common/error_handler.h"
#include "internal.h"
#include <algorithm>
#include <utility>

namespace refactor::graph_topo {

    GraphTopo::GraphTopo()
        : _impl(new __Implement) {}
    GraphTopo::GraphTopo(GraphTopo const &others)
        : _impl(others._impl ? new __Implement(*others._impl) : nullptr) {}
    GraphTopo::GraphTopo(GraphTopo &&others) noexcept
        : _impl(std::exchange(others._impl, nullptr)) {}
    GraphTopo::~GraphTopo() {
        delete std::exchange(_impl, nullptr);
    }

    auto GraphTopo::operator=(GraphTopo const &others) -> GraphTopo & {
        if (this != &others) {
            delete std::exchange(_impl, others._impl ? new __Implement(*others._impl) : nullptr);
        }
        return *this;
    }
    auto GraphTopo::operator=(GraphTopo &&others) noexcept -> GraphTopo & {
        if (this != &others) {
            delete std::exchange(_impl, std::exchange(others._impl, nullptr));
        }
        return *this;
    }

}// namespace refactor::graph_topo
