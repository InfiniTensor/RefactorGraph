#include "runtime/stream.h"

namespace refactor::runtime {

    void *Address::operator()(void *stack) {
        return isBlob()
                   ? std::get<mem_manager::SharedForeignBlob>(value)->ptr()
                   : reinterpret_cast<uint8_t *>(stack) + std::get<size_t>(value);
    }
    bool Address::isBlob() const noexcept {
        return std::holds_alternative<mem_manager::SharedForeignBlob>(value);
    }
    bool Address::isOffset() const noexcept {
        return std::holds_alternative<size_t>(value);
    }

    Stream::Stream(mem_manager::SharedForeignBlob stack,
                   graph_topo::GraphTopo topology,
                   std::vector<_N> routines,
                   std::vector<_E> offsets)
        : _stack(std::move(stack)),
          _internal(_G{
              std::move(topology),
              std::move(routines),
              std::move(offsets),
          }) {}

    void Stream::run() {
        auto stack = _stack->ptr();
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            auto const &routine = _internal.nodes[nodeIdx];
            Addresses
                inputs_(inputs.size()),
                outputs_(outputs.size());
            std::transform(inputs.begin(), inputs.end(),
                           inputs_.begin(),
                           [stack, this](auto i) { return _internal.edges[i](stack); });
            std::transform(outputs.begin(), outputs.end(),
                           outputs_.begin(),
                           [stack, this](auto i) { return _internal.edges[i](stack); });
            routine(_resources, std::move(inputs_), std::move(outputs_));
        }
    }

}// namespace refactor::runtime
