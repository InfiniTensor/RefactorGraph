#include "runtime/stream.h"
#include "runtime/mem_manager.hh"

namespace refactor::runtime {
    using mem_manager::ForeignBlob;

    void emptyRoutine(runtime::Resources &, void *, void const *const *, void *const *) {}

    void *Address::operator()(void *stack) const {
        if (isBlob()) {
            auto blob = std::get<mem_manager::SharedForeignBlob>(value);
            return blob ? (void *) *blob : nullptr;
        }
        return reinterpret_cast<uint8_t *>(stack) + std::get<size_t>(value);
    }
    bool Address::isBlob() const noexcept {
        return std::holds_alternative<mem_manager::SharedForeignBlob>(value);
    }
    bool Address::isOffset() const noexcept {
        return std::holds_alternative<size_t>(value);
    }
    auto Address::blob() const noexcept -> mem_manager::SharedForeignBlob const & {
        return std::get<mem_manager::SharedForeignBlob>(value);
    }
    auto Address::offset() const noexcept -> size_t {
        return std::get<size_t>(value);
    }

    Stream::Stream(Resources resources,
                   size_t stack,
                   std::vector<size_t> outputs,
                   graph_topo::GraphTopo topology,
                   std::vector<_N> routines,
                   std::vector<_E> offsets)
        : _resources(std::move(resources)),
          _stack(ForeignBlob::share(_resources.fetch<MemManager>()->manager, stack)),
          _outputsSize(std::move(outputs)),
          _internal(_G{
              std::move(topology),
              std::move(routines),
              std::move(offsets),
          }) {
    }

    void Stream::setInput(count_t i, void const *data, size_t size) {
        auto globalInputs = _internal.topology.globalInputs();
        ASSERT(i < globalInputs.size(), "input index out of range");

        auto allocator = _resources.fetch<MemManager>()->manager;
        auto blob = ForeignBlob::share(std::move(allocator), size);
        blob->copyIn(data, size);
        _internal.edges[globalInputs[i]].value = {std::move(blob)};
    }
    void Stream::setInput(count_t i, mem_manager::SharedForeignBlob blob) {
        auto globalInputs = _internal.topology.globalInputs();
        ASSERT(i < globalInputs.size(), "input index out of range");

        _internal.edges[globalInputs[i]].value = {std::move(blob)};
    }
    void Stream::getOutput(count_t i, void *data, size_t size) const {
        auto globalOutputs = _internal.topology.globalOutputs();
        ASSERT(i < globalOutputs.size(), "output index out of range");

        _internal.edges[globalOutputs[i]].blob()->copyOut(data, size);
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
            auto allocator = _resources.fetch<MemManager>()->manager;
            auto outputs = _internal.topology.globalOutputs();
            for (auto i : range0_(outputs.size())) {
                _internal.edges[outputs[i]].value = {ForeignBlob::share(allocator, _outputsSize[i])};
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
        auto stack = reinterpret_cast<uint8_t *>(static_cast<void *>(*_stack));
        for (auto const [nodeIdx, i, o] : _internal.topology) {
            auto [inputs, outputs] = collectAddress(*_stack, _internal.edges, buffer, i, o);
            auto const &[routine, workspaceOffset] = _internal.nodes[nodeIdx];
            routine(_resources, stack + workspaceOffset, inputs, outputs);
        }
    }

    auto Stream::bench(void (*sync)()) -> std::vector<std::chrono::nanoseconds> {
        std::vector<void *> buffer(16);
        std::vector<std::chrono::nanoseconds> ans(_internal.nodes.size());
        auto stack = reinterpret_cast<uint8_t *>(static_cast<void *>(*_stack));
        for (auto const [nodeIdx, i, o] : _internal.topology) {
            auto [inputs, outputs] = collectAddress(*_stack, _internal.edges, buffer, i, o);
            auto t0 = std::chrono::high_resolution_clock::now();
            auto const &[routine, workspaceOffset] = _internal.nodes[nodeIdx];
            routine(_resources, stack + workspaceOffset, inputs, outputs);
            if (sync) { sync(); }
            auto t1 = std::chrono::high_resolution_clock::now();
            ans[nodeIdx] = t1 - t0;
        }
        return ans;
    }

    void Stream::trace(std::function<void(count_t, void const *const *, void const *const *)> record) {
        std::vector<void *> buffer(16);
        auto stack = reinterpret_cast<uint8_t *>(static_cast<void *>(*_stack));
        for (auto const [nodeIdx, i, o] : _internal.topology) {
            auto [inputs, outputs] = collectAddress(*_stack, _internal.edges, buffer, i, o);
            auto const &[routine, workspaceOffset] = _internal.nodes[nodeIdx];
            routine(_resources, stack + workspaceOffset, inputs, outputs);
            record(nodeIdx, inputs, outputs);
        }
    }

}// namespace refactor::runtime
