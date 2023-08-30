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

    GraphTopo GraphTopo::__withGlobalInputs(size_t globalInputsCount) {
        GraphTopo ans;
        ans._impl->_globalInputsCount = globalInputsCount;
        return ans;
    }
    void GraphTopo::__addNode(size_t newLocalEdgesCount, std::vector<size_t> inputs, size_t outputsCount) {
        _impl->_nodes.push_back({newLocalEdgesCount, inputs.size(), outputsCount});
        for (auto const &edge : inputs) {
            _impl->_connections.push_back(OutputEdge{edge});
        }
    }
    void GraphTopo::__setGlobalOutputs(std::vector<size_t> outputs) {
        for (auto const &edge : outputs) {
            _impl->_connections.push_back(OutputEdge{edge});
        }
    }

}// namespace refactor::graph_topo
