////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HELLM/HEMMer.hpp"
#include "HELLM/ModelArgs.hpp"
#include "HELLM/TorchTransformerBlock.hpp"
#include "HELLM/TransformerBlock.hpp"

#include "HEaaN/device/CudaTools.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sentencepiece_processor.h>
#include <stdexcept>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>

#include <mpi.h>

namespace {


std::string toUpper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return std::toupper(ch); });
    return value;
}

std::string getEnvString(const char *name, const std::string &default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || std::string(value).empty()) {
        return default_value;
    }
    return value;
}

int getEnvInt(const char *name, int default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || std::string(value).empty()) {
        return default_value;
    }
    return std::stoi(value);
}

struct EvalConfig {
    std::string name;
    std::string data_path;
    std::string eval_container;
    int default_eval_steps;
};

EvalConfig getEvalConfig() {
    std::string task = toUpper(getEnvString("HELLM_TASK", "RTE"));
    if (task == "R") {
        task = "RTE";
    } else if (task == "M") {
        task = "MRPC";
    } else if (task == "C") {
        task = "COLA";
    } else if (task == "T") {
        task = "SST2";
    } else if (task == "Q") {
        task = "QNLI";
    }

    if (task == "RTE") {
        return {"RTE", "./data_2ly_rte/", "converted_weights_rte_eval.pth", 277};
    }
    if (task == "MRPC") {
        return {"MRPC", "./data_2ly_mrpc/", "converted_weights_mrpc_eval.pth", 408};
    }
    if (task == "COLA") {
        return {"COLA", "./data_2ly_cola/", "converted_weights_cola_eval.pth", 1043};
    }
    if (task == "SST2") {
        return {"SST2", "./data_2ly_sst2/", "converted_weights_sst2_eval.pth", 872};
    }
    if (task == "QNLI") {
        return {"QNLI", "./data_2ly_qnli/", "converted_weights_qnli_eval.pth", 5463};
    }

    throw std::invalid_argument(
        "Unsupported HELLM_TASK. Use RTE, MRPC, COLA, SST2, or QNLI.");
}

std::string withTrailingSlash(std::string path) {
    if (!path.empty() && path.back() != '/') {
        path += '/';
    }
    return path;
}

} // namespace

int main(int argc, char *argv[]) {
    static const std::string LORA_TYPE = "qkv";
    // static const std::string LORA_TYPE = "";

    const EvalConfig config = getEvalConfig();
    const std::string data_path =
        withTrailingSlash(getEnvString("HELLM_WEIGHT_PATH", config.data_path));
    const int eval_steps = getEnvInt("HELLM_EVAL_STEPS", config.default_eval_steps);
    std::string lora_path = data_path;
    if (argc > 1) {
        lora_path = withTrailingSlash(argv[1]);
    }
    lora_path = withTrailingSlash(getEnvString("HELLM_LORA_PATH", lora_path));

    auto *hemmer = new HELLM::HEMMer{HELLM::HEMMer::genHEMMerMultiGPU()};
    hemmer->setWeightPath(lora_path);
    hemmer->setWeightTestPath(lora_path);
    MPI_Barrier(MPI_COMM_WORLD);

    HELLM::TransformerBlock block{hemmer, data_path, data_path, 0};
    int rank = hemmer->getRank();
    int size = hemmer->getMaxRank();
    std::cout << "rank " << rank << " max rank " << size << std::endl;
    if (rank == 0) {
        std::cout << "task: " << config.name << std::endl;
        std::cout << "eval data path: " << data_path << std::endl;
        std::cout << "eval container: " << config.eval_container << std::endl;
        std::cout << "LoRA snapshot path: " << lora_path << std::endl;
        std::cout << "eval steps: " << eval_steps << std::endl;
    }

    int prompt_len = 128;
    auto container = torch::jit::load(data_path + config.eval_container);

    /* if (rank == 0) {
        for (int i = 0 ; i < 2 ; ++i) {
            std::shared_ptr<HELLM::TorchTransformerBlock> torch_block;
            torch_block = std::make_shared<HELLM::TorchTransformerBlock>(
                    hemmer, container, i);
            torch_block->generateInitialLoraWeight(LORA_TYPE);
        }
    }
    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD); */

    // MRPC:51
    // RTE: 34 + 5
    // COLA: 130 + 3
    // QNLI: 682 + 7
    for (int j = 0; j < eval_steps; ++j) {
        auto inp = torch::empty({prompt_len, HELLM::ModelArgs::DIM});
        auto input_name = "input_" + std::to_string(rank + size * j);
        auto tok_emb_w = container.attr(input_name).toTensor().transpose(0, 1);
        // std::cout << "dim = " << tok_emb_w.sizes() << std::endl;

        auto mask_name = "mask_" + std::to_string(rank + size * j);
        auto mask = container.attr(mask_name).toTensor();
        mask = mask.squeeze().slice(0, 0).unsqueeze(0).repeat({128, 1});
        for (int i = 0; i < prompt_len; ++i) {
            inp[i] = tok_emb_w[0][i];
        }
        inp = inp.view({prompt_len, HELLM::ModelArgs::N_HEAD,
                        HELLM::ModelArgs::HEAD_DIM})
                  .transpose(0, 1);
        std::vector<torch::Tensor> cur{HELLM::ModelArgs::N_HEAD};
        std::vector<HELLM::CtxtTensor> ctxt_cur;
        ctxt_cur.reserve(HELLM::ModelArgs::N_HEAD / 2);

        for (long i = 0; i < HELLM::ModelArgs::N_HEAD / 2; ++i) {
            auto input_ctxt = hemmer->encrypt2(inp[i * 2], inp[i * 2 + 1]);
            ctxt_cur.push_back(input_ctxt);
        }

        HEaaN::CudaTools::cudaDeviceSynchronize();

        auto exp_mask = hemmer->message2(mask, mask);

        for (int i = 0; i < 2; ++i) {

            HELLM::TransformerBlock block_tmp{hemmer, data_path, data_path, i};

            ctxt_cur =
                block_tmp.forward2_bert_eval(ctxt_cur, exp_mask, LORA_TYPE);
            // ctxt_cur = block_tmp.forward2_bert_SM(ctxt_cur, LORA_TYPE);
            // ctxt_cur = block_tmp.forward2_bert_test(ctxt_cur, exp_mask,
            // LORA_TYPE);
            HEaaN::CudaTools::cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
        }
        block.forward3_pooling_bert_test(ctxt_cur);
        HEaaN::CudaTools::cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    delete hemmer;
    HELLM::cleanUpMatrixTransformer();
}
