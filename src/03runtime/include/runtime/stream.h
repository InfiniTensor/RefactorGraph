#ifndef RUNTIME_STREAM_H
#define RUNTIME_STREAM_H

#include "graph_topo/graph_topo.h"
#include "mem_manager/foreign_blob.hh"
#include "resource.h"
#include <absl/container/inlined_vector.h>
#include <functional>

namespace refactor::runtime {
    using Addresses = absl::InlinedVector<void *, 2>;
    using Routine = std::function<void(runtime::Resources &, Addresses, Addresses)>;

    class Stream {
        using _N = Routine;
        using _E = size_t;
        using _G = graph_topo::Graph<_N, _E>;

        Resources _resources;
        mem_manager::SharedForeignBlob _stack;
        _G _internal;

    public:
        Stream(mem_manager::SharedForeignBlob,
               graph_topo::GraphTopo,
               std::vector<_N>,
               std::vector<_E>);
        void run();
    };

}// namespace refactor::runtime

#endif// RUNTIME_STREAM_H
