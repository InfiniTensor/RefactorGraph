#include "runtime/stream.h"

namespace refactor::runtime {
    using namespace hardware;

    void emptyRoutine(runtime::Resources &, void *, void const *const *, void *const *) {}

    void *Address::operator()(void *stack) const {
        if (isBlob()) {
            auto blob = std::get<Arc<Device::Blob>>(value);
            return blob ? blob->get() : nullptr;
        }
        return reinterpret_cast<uint8_t *>(stack) + std::get<size_t>(value);
    }
    bool Address::isBlob() const noexcept {
        return std::holds_alternative<Arc<Device::Blob>>(value);
    }
    bool Address::isOffset() const noexcept {
        return std::holds_alternative<size_t>(value);
    }
    auto Address::blob() const noexcept -> Device::Blob const & {
        return *std::get<Arc<Device::Blob>>(value);
    }
    auto Address::offset() const noexcept -> size_t {
        return std::get<size_t>(value);
    }

    Stream::Stream(decltype(_resources) resources,
                   decltype(_device) device,
                   decltype(_stackSize) stack,
                   decltype(_outputsSize) outputs,
                   graph_topo::GraphTopo topology,
                   std::vector<_N> routines,
                   std::vector<_E> offsets)
        : _resources(std::move(resources)),
          _device(std::move(device)),
          _stackSize(stack),
          _outputsSize(std::move(outputs)),
          _internal(_G{
              std::move(topology),
              std::move(routines),
              std::move(offsets),
          }),
          _stack(nullptr) {}

    void Stream::setData(count_t i, void const *data, size_t size) {
        auto blob = _device->malloc(size);
        blob->copyFromHost(data, size);
        _internal.edges[i].value = {std::move(blob)};
    }
    void Stream::setData(count_t i, Arc<Device::Blob> blob) {
        _internal.edges[i].value = {std::move(blob)};
    }
    void Stream::getData(count_t i, void *data, size_t size) const {
        _internal.edges[i].blob().copyToHost(data, size);
    }

    auto Stream::prepare() -> std::vector<count_t> {
        auto globalInputs = _internal.topology.globalInputs();
        std::vector<count_t> unknownInputs;
        for (auto i : range0_(globalInputs.size())) {
            if (!_internal.edges[globalInputs[i]].blob()) {
                unknownInputs.push_back(i);
            }
        }
        if (unknownInputs.empty()) {
            auto outputs = _internal.topology.globalOutputs();
            for (auto i : range0_(outputs.size())) {
                _internal.edges[outputs[i]].value = {_device->malloc(_outputsSize[i])};
            }
            if (!_stack) {
                _stack = _device->malloc(_stackSize);
            }
        }
        return unknownInputs;
    }

    template<class I, class O>
    std::pair<void const **, void **> collectAddress(
        void *stack,
        std::vector<Address> const &addresses,
        std::vector<void *> &buffer,
        I i, O o) {
        buffer.resize(i.size() + o.size());
        auto inputs = buffer.data(),
             outputs = std::transform(i.begin(), i.end(), inputs, [&](auto i) { return addresses[i](stack); });
        /* alignnnnn*/ std::transform(o.begin(), o.end(), outputs, [&](auto i) { return addresses[i](stack); });
        return {const_cast<void const **>(inputs), outputs};
    }

    void Stream::run() {
        std::vector<void *> buffer(16);
        auto stack = _stack->get<uint8_t>();
        for (auto const [nodeIdx, i, o] : _internal.topology) {
            auto [inputs, outputs] = collectAddress(stack, _internal.edges, buffer, i, o);
            auto const &[routine, workspaceOffset] = _internal.nodes[nodeIdx];
            _device->setContext();
            routine(_resources, stack + workspaceOffset, inputs, outputs);
        }
    }

    auto Stream::bench(void (*sync)()) -> std::vector<std::chrono::nanoseconds> {
        std::vector<void *> buffer(16);
        std::vector<std::chrono::nanoseconds> ans(_internal.nodes.size());
        auto stack = _stack->get<uint8_t>();
        for (auto const [nodeIdx, i, o] : _internal.topology) {
            auto [inputs, outputs] = collectAddress(stack, _internal.edges, buffer, i, o);
            auto t0 = std::chrono::high_resolution_clock::now();
            auto const &[routine, workspaceOffset] = _internal.nodes[nodeIdx];
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
        for (auto const [nodeIdx, i, o] : _internal.topology) {
            auto [inputs, outputs] = collectAddress(stack, _internal.edges, buffer, i, o);
            auto const &[routine, workspaceOffset] = _internal.nodes[nodeIdx];
            _device->setContext();
            routine(_resources, stack + workspaceOffset, inputs, outputs);
            record(nodeIdx, inputs, outputs);
        }
    }

}// namespace refactor::runtime
