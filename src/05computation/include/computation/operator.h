#ifndef COMPUTATION_OPERATOR_H
#define COMPUTATION_OPERATOR_H

#include "kernel/collector.h"
#include "kernel/target.h"

namespace refactor::computation {
    using kernel::LayoutType;
    using kernel::Shape;
    using kernel::Target;
    using kernel::Tensor;

    class Operator {
    public:
        virtual ~Operator() = default;
        virtual size_t opTypeId() const = 0;
        virtual std::string_view name() const = 0;
        virtual bool isLayoutDependent() const;
        virtual bool isIdentity() const;
        virtual void transposeTo(LayoutType);
        virtual kernel::CollectorBox candidateKernels(Target) const;

        template<class T, class... Args>
        bool is(Args &&...args) const noexcept {
            return this->opTypeId() == T::typeId(std::forward<Args>(args)...);
        }
    };

    class LayoutDependentOperator : public Operator {
    public:
        virtual bool isLayoutDependent() const final;
        virtual void transposeTo(LayoutType) final;
    };

    class AxisRankOperator : public Operator {
    public:
        uint32_t axis, rank;

        constexpr AxisRankOperator(uint32_t axis_, uint32_t rank_)
            : Operator(), axis(axis_), rank(rank_) {}
        bool isLayoutDependent() const final;
        void transposeTo(LayoutType target) final;
    };

    using OpBox = std::unique_ptr<Operator>;

    struct MyOperator {
        size_t numInputs;
        Arc<Operator> base;

        MyOperator(size_t num) : numInputs(num) {}
        virtual ~MyOperator() = default;
    };

    struct MyOperator_U : public MyOperator {
        MyOperator_U() : MyOperator(1) {}

        virtual bool compute(Tensor const &, Tensor &) const = 0;
        virtual Shape verify(Tensor const &) const = 0;
    };

    struct MyOperator_B : public MyOperator {
        MyOperator_B() : MyOperator(2) {}

        virtual bool compute(Tensor const &, Tensor const &, Tensor &) const = 0;
        virtual Shape verify(Tensor const &, Tensor const &) const = 0;
    };

}// namespace refactor::computation

#endif// COMPUTATION_OPERATOR_H
