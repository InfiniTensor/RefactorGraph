#include "computation/operators/simple_binary.h"

namespace refactor::computation {
    using Op = SimpleBinary;
    using Ty = SimpleBinaryType;
    using Collector_ = kernel::SimpleBinaryCollector;

    auto Op::typeId(Ty type) noexcept -> size_t {
        switch (type) {
            case Ty::Add: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Sub: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Mul: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Div: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Pow: {
                static uint8_t ID = 5;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::And: {
                static uint8_t ID = 6;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Or: {
                static uint8_t ID = 7;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Xor: {
                static uint8_t ID = 8;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Mod: {
                static uint8_t ID = 9;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Fmod: {
                static uint8_t ID = 10;
                return reinterpret_cast<size_t>(&ID);
            }            
            default:
                UNREACHABLE();
        }
    }
    auto Op::opTypeId() const noexcept -> size_t {
        return typeId(type);
    }
    auto Op::name() const noexcept -> std::string_view {
        switch (type) {
            case Ty::Add:
                return "Add";
            case Ty::Sub:
                return "Sub";
            case Ty::Mul:
                return "Mul";
            case Ty::Div:
                return "Div";
            case Ty::Pow:
                return "Pow";
            case Ty::And:
                return "And";
            case Ty::Or:
                return "Or";
            case Ty::Xor:
                return "Xor";
            case Ty::Mod:
                return "Mod";
            case Ty::Fmod:
                return "Fmod";                
            default:
                UNREACHABLE();
        }
    }

    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        return std::make_unique<Collector_>(target, type);
    }
    auto Op::serialize() const noexcept -> std::string {
        return fmt::format("{}()", name());
    }

}// namespace refactor::computation
