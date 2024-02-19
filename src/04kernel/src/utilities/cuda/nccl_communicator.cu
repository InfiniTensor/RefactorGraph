#include "common.h"
#include "nccl_communicator.hh"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>


namespace refactor::kernel::nccl {
    NcclCommunicator::NcclCommunicator(int worldSize, int rank) : worldSize_(worldSize), rank_(rank) {
        const std::string filePath("./nccl_id.bin");

        ncclUniqueId commId;

        if (rank == 0) {
            checkNcclError(ncclGetUniqueId(&commId));
            std::ofstream ofs(filePath, std::ios::binary);
            ofs.write((char *) &commId, sizeof(ncclUniqueId));

        } else {
            auto begin = std::chrono::steady_clock::now();
            while (!std::filesystem::exists(filePath)) {
                auto now = std::chrono::steady_clock::now();
                ASSERT(now < begin + std::chrono::seconds(10),
                       "time limit (10s) exceeded.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::ifstream ifs(filePath, std::ios::binary);
            ifs.read((char *) &commId, sizeof(ncclUniqueId));
        }
        checkNcclError(ncclCommInitRank(&comm, worldSize, commId, rank));

        if (rank == 0) {
            std::filesystem::remove(filePath);
        }

        printf("Rank %d established NCCL communicator.\n", rank);
    }

    NcclCommunicator::~NcclCommunicator() {
        checkNcclError(ncclCommFinalize(comm));
        checkNcclError(ncclCommDestroy(comm));
    }

    auto NcclCommunicator::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto NcclCommunicator::build(int worldSize, int rank) noexcept -> runtime::ResourceBox {
        return std::make_unique<NcclCommunicator>(worldSize, rank);
    }

    auto NcclCommunicator::resourceTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto NcclCommunicator::description() const noexcept -> std::string_view {
        return "NcclCommunicator";
    }

}// namespace refactor::kernel::nccl
