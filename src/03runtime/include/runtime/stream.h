#ifndef RUNTIME_STREAM_H
#define RUNTIME_STREAM_H

#include "graph_topo.h"
#include "mem_manager/foreign_blob.hh"
#include "resource.h"
#include <absl/container/inlined_vector.h>
#include <chrono>
#include <functional>
#include <variant>

namespace refactor::runtime {
    using Routine = std::function<void(runtime::Resources &, void const **, void **)>;

    void emptyRoutine(runtime::Resources &, void const **, void **);

    struct Address {
        std::variant<size_t, mem_manager::SharedForeignBlob> value;

        void *operator()(void *stack) const;

        bool isBlob() const noexcept;
        bool isOffset() const noexcept;

        auto blob() const noexcept -> mem_manager::SharedForeignBlob const &;
        auto offset() const noexcept -> size_t;
    };

    class Stream {
        using _N = Routine;
        using _E = Address;
        using _G = graph_topo::Graph<_N, _E>;

        Resources _resources;
        mem_manager::SharedForeignBlob _stack;
        std::vector<size_t> _outputsSize;
        _G _internal;

    public:
        Stream(Resources,
               size_t stack,
               std::vector<size_t> outputs,
               graph_topo::GraphTopo,
               std::vector<_N>,
               std::vector<_E>);
        void setInput(count_t, void const *, size_t);
        void setInput(count_t, mem_manager::SharedForeignBlob);
        void getOutput(count_t, void *, size_t) const;
        auto prepare() -> std::vector<count_t>;
        void run();
        auto bench(void (*sync)()) -> std::vector<std::chrono::nanoseconds>;
        void trace(std::function<void(count_t, void const *const *, void const *const *)>);
    };

}// namespace refactor::runtime

#endif// RUNTIME_STREAM_H
