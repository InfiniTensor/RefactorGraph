#include "runtime/stream.h"
#include "runtime/mem_manager.hh"

namespace refactor::runtime {
    using mem_manager::ForeignBlob;

    void emptyRoutine(runtime::Resources &, void const **, void **) {}

    void *Address::operator()(void *stack) {
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

    void Stream::setInput(uint_lv1 i, void const *data, size_t size) {
        auto globalInputs = _internal.topology.globalInputs();
        ASSERT(i < globalInputs.size(), "input index out of range");

        auto allocator = _resources.fetch<MemManager>()->manager;
        auto blob = ForeignBlob::share(std::move(allocator), size);
        blob->copyIn(data, size);
        _internal.edges[globalInputs[i]].value = {std::move(blob)};
    }
    void Stream::setInput(uint_lv1 i, mem_manager::SharedForeignBlob blob) {
        auto globalInputs = _internal.topology.globalInputs();
        ASSERT(i < globalInputs.size(), "input index out of range");

        _internal.edges[globalInputs[i]].value = {std::move(blob)};
    }
    void Stream::getOutput(uint_lv1 i, void *data, size_t size) const {
        auto globalOutputs = _internal.topology.globalOutputs();
        ASSERT(i < globalOutputs.size(), "input index out of range");

        _internal.edges[globalOutputs[i]].blob()->copyOut(data, size);
    }

    std::vector<uint_lv1> Stream::prepare() {
        auto globalInputs = _internal.topology.globalInputs();
        std::vector<uint_lv1> unknownInputs;
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

    void Stream::run() {
        auto map = [this](auto i) { return _internal.edges[i](*_stack); };
        std::vector<void *> buffer(16);
        for (auto const [nodeIdx, i, o] : _internal.topology) {
            buffer.resize(i.size() + o.size());
            auto inputs_ = buffer.data(),
                 outputs_ = std::transform(i.begin(), i.end(), inputs_, map);
            /* alignment */ std::transform(o.begin(), o.end(), outputs_, map);
            _internal.nodes[nodeIdx](_resources, const_cast<void const **>(inputs_), outputs_);
        }
    }

}// namespace refactor::runtime
