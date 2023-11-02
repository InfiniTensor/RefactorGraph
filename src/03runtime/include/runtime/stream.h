#ifndef RUNTIME_STREAM_H
#define RUNTIME_STREAM_H

#include "graph_topo.h"
#include "mem_manager/foreign_blob.hh"
#include "resource.h"
#include <absl/container/inlined_vector.h>
#include <functional>
#include <variant>

namespace refactor::runtime {
    using Routine = std::function<void(runtime::Resources &, void const **, void **)>;

    void emptyRoutine(runtime::Resources &, void const **, void **);

    struct Address {
        std::variant<size_t, mem_manager::SharedForeignBlob> value;

        void *operator()(void *stack);

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
        void setInput(uint_lv1, void const *, size_t);
        void setInput(uint_lv1, mem_manager::SharedForeignBlob);
        std::vector<uint_lv1> prepare();
        void run();
    };

}// namespace refactor::runtime

#endif// RUNTIME_STREAM_H
