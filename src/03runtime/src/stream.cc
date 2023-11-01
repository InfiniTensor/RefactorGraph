#include "runtime/stream.h"

namespace refactor::runtime {

    void emptyRoutine(runtime::Resources &, void const **, void **) {}

    void *Address::operator()(void *stack) {
        return isBlob()
                   ? *std::get<mem_manager::SharedForeignBlob>(value)
                   : reinterpret_cast<uint8_t *>(stack) + std::get<size_t>(value);
    }
    bool Address::isBlob() const noexcept {
        return std::holds_alternative<mem_manager::SharedForeignBlob>(value);
    }
    bool Address::isOffset() const noexcept {
        return std::holds_alternative<size_t>(value);
    }

    size_t Address::getOffset() const {
        return std::get<size_t>(value);
    }

    Stream::Stream(Resources resources,
                   mem_manager::SharedForeignBlob stack,
                   graph_topo::GraphTopo topology,
                   std::vector<_N> routines,
                   std::vector<_E> offsets)
        : _resources(std::move(resources)),
          _stack(std::move(stack)),
          _internal(_G{
              std::move(topology),
              std::move(routines),
              std::move(offsets),
          }) {}

    void Stream::run() {
        auto map = [this](auto i) { return _internal.edges[i](*_stack); };
        std::vector<void *> buffer(16);
        for (auto const [nodeIdx, i, o] : _internal.topology) {
            buffer.resize(i.size() + o.size());
            auto inputs_ = buffer.data(),
                 outputs_ = std::transform(i.begin(), i.end(), inputs_, map);
            /* alignnnnn */ std::transform(o.begin(), o.end(), outputs_, map);
            _internal.nodes[nodeIdx](_resources, const_cast<void const **>(inputs_), outputs_);
        }
    }

}// namespace refactor::runtime
