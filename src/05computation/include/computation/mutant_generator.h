#ifndef COMPUTATION_MUTANT_GENERATOR_H
#define COMPUTATION_MUTANT_GENERATOR_H

#include "graph_mutant.h"
#include "operator.h"

namespace refactor::computation {

    using OpVec = std::vector<Arc<MyOperator>>;
    using TensorVec = std::vector<Rc<refactor::graph_topo::LinkedGraph<Node, Edge>::Edge>>;

    inline uint64_t hashAppend(uint64_t a, uint64_t b) {
        return (a * 10000019 + b * 10000079) % 2147483647;
    }

    template<typename T> inline uint64_t hashVector(const std::vector<T> &vec) {
        uint64_t ret = 0;
        for (auto v : vec)
            ret = hashAppend(ret, v);
        return ret;
    }

    class MutantGenerator {
        float equalThreshold;
        size_t maxDepth;
        size_t numValidTensors = 0;
        OpVec opList;
        std::vector<OpVec> opStorage;
        OpVec opFinger;
        TensorVec validTensors;
        std::set<uint64_t> opHashMaps;

    public:
        void init(float, size_t, OpVec) noexcept;
        void run(GraphMutant const &, std::vector<GraphMutant> &) noexcept;
        void dfs(size_t, GraphMutant const &, GraphMutant &, std::vector<GraphMutant> &) noexcept;
        bool is_mutant(GraphMutant &, GraphMutant const &) noexcept;
        bool approx_equal(const Tensor &, const Tensor &) const noexcept;
        bool have_same_op(Arc<MyOperator> const &, size_t, size_t) noexcept;
        void delete_hash_op(Arc<MyOperator> const &, size_t, size_t) noexcept;
    };
}// namespace refactor::computation

#endif// COMPUTATION_MUTANT_GENERATOR_H