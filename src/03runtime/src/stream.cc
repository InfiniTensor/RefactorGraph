#include "runtime/stream.h"

namespace refactor::runtime {
    void emptyRoutine(runtime::Resources &, void *, void const *const *, void *const *) {}

    Stream::Stream(decltype(_resources) resources,
                   size_t stackSize,
                   graph_topo::GraphTopo topology,
                   std::vector<Node> nodes,
                   std::vector<Edge> edges,
                   decltype(_device) device)
        : _device(std::move(device)),
          _stack(_device->malloc(stackSize)),
          _resources(std::move(resources)),
          _graph{
              std::move(topology),
              std::move(nodes),
              std::move(edges),
          } {}

    void Stream::setData(count_t i, void const *data, size_t size) {
        auto blob = _device->malloc(size);
        blob->copyFromHost(data, size);
        _graph.edges[i].blob = std::move(blob);
    }
    void Stream::setData(count_t i, Arc<hardware::Device::Blob> blob) {
        _graph.edges[i].blob = std::move(blob);
    }
    bool Stream::getData(count_t i, void *data, size_t size) const {
        if (!_graph.edges[i].blob) { return false; }
        _graph.edges[i].blob->copyToHost(data, size);
        return true;
    }

    Resources& Stream::getResources() {
        return _resources;
    }

    template<class I, class O>
    std::pair<void const **, void **> collectAddress(
        void *stack,
        std::vector<Edge> const &edges,
        std::vector<void *> &buffer,
        I i, O o) {
        auto fn = [&](auto i) -> void * {
            if (edges[i].blob) {
                return edges[i].blob->get();
            }
            return reinterpret_cast<uint8_t *>(stack) + edges[i].stackOffset;
        };
        buffer.resize(i.size() + o.size());
        auto inputs = buffer.data(),
             outputs = std::transform(i.begin(), i.end(), inputs, fn);
        /* alignnnnn*/ std::transform(o.begin(), o.end(), outputs, fn);
        return {const_cast<void const **>(inputs), outputs};
    }

    void Stream::run() {
        std::vector<void *> buffer(16);
        auto stack = _stack->get<uint8_t>();
        for (auto const [nodeIdx, i, o] : _graph.topology) {
            auto [inputs, outputs] = collectAddress(stack, _graph.edges, buffer, i, o);
            auto const &[routine, workspaceOffset] = _graph.nodes[nodeIdx];
            _device->setContext();
            routine(_resources, stack + workspaceOffset, inputs, outputs);
        }
    }

    auto Stream::bench(void (*sync)()) -> std::vector<std::chrono::nanoseconds> {
        std::vector<void *> buffer(16);
        std::vector<std::chrono::nanoseconds> ans(_graph.nodes.size());
        auto stack = _stack->get<uint8_t>();
        for (auto const [nodeIdx, i, o] : _graph.topology) {
            auto [inputs, outputs] = collectAddress(stack, _graph.edges, buffer, i, o);
            auto t0 = std::chrono::high_resolution_clock::now();
            auto const &[routine, workspaceOffset] = _graph.nodes[nodeIdx];
            _device->setContext();
            routine(_resources, stack + workspaceOffset, inputs, outputs);
            if (sync) { sync(); }
            auto t1 = std::chrono::high_resolution_clock::now();
            ans[nodeIdx] = t1 - t0;
        }
        return ans;
    }

    void Stream::trace(std::function<void(count_t, void const *const *, void const *const *)> record) {
        std::vector<void *> buffer(16);
        auto stack = _stack->get<uint8_t>();
        for (auto const [nodeIdx, i, o] : _graph.topology) {
            auto [inputs, outputs] = collectAddress(stack, _graph.edges, buffer, i, o);
            auto const &[routine, workspaceOffset] = _graph.nodes[nodeIdx];
            _device->setContext();
            routine(_resources, stack + workspaceOffset, inputs, outputs);
            record(nodeIdx, inputs, outputs);
        }
    }

}// namespace refactor::runtime
