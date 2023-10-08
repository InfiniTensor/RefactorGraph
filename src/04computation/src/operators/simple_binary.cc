#include "computation/operators/simple_binary.h"

namespace refactor::computation {

    size_t SimpleBinary::typeId(SimpleBinaryType type) noexcept {
        switch (type) {
            case SimpleBinaryType::Add: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleBinaryType::Sub: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleBinaryType::Mul: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleBinaryType::Div: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleBinaryType::Pow: {
                static uint8_t ID = 5;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleBinaryType::And: {
                static uint8_t ID = 6;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleBinaryType::Or: {
                static uint8_t ID = 7;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleBinaryType::Xor: {
                static uint8_t ID = 8;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }
    size_t SimpleBinary::opTypeId() const noexcept {
        return typeId(type);
    }
    std::string_view SimpleBinary::name() const noexcept {
        switch (type) {
            case SimpleBinaryType::Add:
                return "Add";
            case SimpleBinaryType::Sub:
                return "Sub";
            case SimpleBinaryType::Mul:
                return "Mul";
            case SimpleBinaryType::Div:
                return "Div";
            case SimpleBinaryType::Pow:
                return "Pow";
            case SimpleBinaryType::And:
                return "And";
            case SimpleBinaryType::Or:
                return "Or";
            case SimpleBinaryType::Xor:
                return "Xor";
            default:
                UNREACHABLE();
        }
    }

    kernel::CollectorBox SimpleBinary::candidateKernels() const noexcept {
        return std::make_unique<kernel::SimpleBinaryCollector>(type);
    }

}// namespace refactor::computation
