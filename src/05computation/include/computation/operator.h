#ifndef COMPUTATION_OPERATOR_H
#define COMPUTATION_OPERATOR_H

#include "common/error_handler.h"
#include "kernel/collector.h"
#include "kernel/target.h"

namespace refactor::computation {
    using kernel::LayoutType;
    using kernel::Target;

    class Operator {
    public:
        virtual size_t opTypeId() const = 0;
        virtual std::string_view name() const = 0;
        virtual bool isLayoutDependent() const;
        virtual void transposeTo(LayoutType);
        virtual kernel::CollectorBox candidateKernels(Target) const;

        template<class T>
        bool is() const {
            return opTypeId() == T::typeId();
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

}// namespace refactor::computation

#endif// COMPUTATION_OPERATOR_H
