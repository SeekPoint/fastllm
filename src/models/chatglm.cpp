//
// Created by huangyuyang on 5/11/23.
//

#include "utils.h"

#include "chatglm.h"

#include <cmath>

#include <chrono>

#include <algorithm>

#include <map>

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    void ChatGLMModel::UpdateSinCos(float rope) {
        if (rope == this->rope) {
            return;
        }
        this->rope = rope;

        // 这两行代码调整 sin 和 cos 向量的大小以匹配 max_positions。
        sin.resize(max_positions);
        cos.resize(max_positions);
        std::vector <float> invFreq;

        // 这部分代码计算了一系列频率的倒数，并将结果存储在 invFreq 向量中。
        for (int i = 0; i < rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(10000, (float)i / rotary_dim));
        }

        // 使用这些倒数 和 位置索引 i 来计算正弦和余弦值，并将这些值存储在 sin 和 cos 向量中。
        for (int i = 0; i < max_positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size(); j++) {
                sin[i][j] = ::sin((float)i / rope * invFreq[j]);
                cos[i][j] = ::cos((float)i / rope * invFreq[j]);
            }
        }

        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            for (int j = 0; j < sin[0].size(); j++) {
                fsin.push_back(sin[i][j]);
                fcos.push_back(cos[i][j]);
            }
        }

        // 这两行代码将 sin 和 cos 向量中的数据复制到 sinData 和 cosData 对象中。
        sinData.CopyFrom(Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, fsin));
        cosData.CopyFrom(Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, fcos));
    }

    ChatGLMModel::ChatGLMModel() {
        this->model_type = "chatglm"; //这行代码设置类的 model_type 成员变量为 "chatglm"。this 是一个指向当前对象的指针。

        this->bos_token_id = 130004; // 设置句子的开始标记
        this->eos_token_id = 130005; // 设置句子的结束标记
        this->rope = -1.0;
        this->UpdateSinCos(1.0f);

        // 设置 weight.embeddingNames 成员变量的值。
        weight.embeddingNames.insert("transformer.word_embeddings.weight");
        weight.embeddingNames.insert("transformer.embedding.word_embeddings.weight");
    }

    int ChatGLMModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                              const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                              const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                              std::vector <float> *logits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(logits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    /*
    ChatGLMModel::ForwardBatch 函数解析
    我在下面的代码里面逐步添加了一些注释，这里的c++代码都对应了huggingface上面的chatglm-6b和chatglm2-6b的模型定义和推理的代码。
    我不仅加了注释，还把中间tensor的维度变化也标注出来了，对KV Cache的实现也加了解释。
    我在这里发现了唯一一个和python代码对不上的问题是在self attention的softmax之前有一行Mul(attnProbs, i + 1, attnProbs);
    ，这行代码我不是很确定作用，我猜测是让attnProbs更大一些来降低数值溢出的风险？
    在huggingface的实现中是不存在这一行代码的。
    */
    //// 这个函数是 ChatGLMModel 类的一个成员函数，名为 ForwardBatch。用于处理一批数据的前向传播。
    std::vector <int> ChatGLMModel::ForwardBatch(
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector <std::pair <Data, Data> > &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector <std::vector <float>*> *retLogits) {
        if (this->weight.dicts.find("rope_ratio") != this->weight.dicts.end()) {
            UpdateSinCos(atof(this->weight.dicts["rope_ratio"].c_str()));
        }

         // 获取 inputIds 的第二个维度大小，存储在 maxLen 中，代表输入序列的最大长度。
        int maxLen = inputIds.dims[1];

        // 声明了一系列 Data 类型的变量，这些变量用于存储中间计算结果。
        Data inputEmbeddings;
        Data attenInput;
        Data qkv, q, k, v;
        Data attnProbs;
        Data attnOutput;
        Data contextLayer;
        Data mlpInput;
        Data middle, middle2;
        Data temp;

        // 定义一个整型向量，可能用于存储最后的返回结果。
        std::vector<int> lastRet;

        // ChatGLMBlock
        // 调用 GetVersion 函数获取模型的版本。
        int version = GetVersion();

        // 根据版本设置 weightPre 和 weightMiddle 的值，这两个变量会被用于构造权重的名称。
        std::string weightPre, weightMiddle;

        if (version == 1) {
            weightPre = "transformer.layers.";
            weightMiddle = ".attention";
        } else if (version == 2) {
            weightPre = "transformer.encoder.layers.";
            weightMiddle = ".self_attention";
        }

        // ChatGLM2
        // 定义一个 Data 类型的变量 inputIdsPermute，用于存储置换后的输入 ID。
        Data inputIdsPermute;

        // 对 inputIds 进行置换，将batch维度和序列长度维度交换。
        //【bs, seq_length, hidden_size】->【seq_length, bs, hidden_size】
        Permute(inputIds, {1, 0}, inputIdsPermute);

        // 调用 Embedding 函数，对输入的单词 ID 进行嵌入，生成 inputEmbeddings。
        Embedding(inputIdsPermute, this->weight["transformer" + std::string((version == 2 ? ".embedding" : "")) +
                                                ".word_embeddings.weight"], inputEmbeddings);

        // 定义一个引用 hiddenStates，指向 inputEmbeddings，
        // 在后续的操作中，对 hiddenStates 的修改也会改变 inputEmbeddings。
        Data &hiddenStates = inputEmbeddings;

        // 针对每个 Transformer 的 block 进行操作。在每个循环中：
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);

            // 首先，使用 LayerNorm 或 RMSNorm 对 hiddenStates 进行归一化操作，得到 attenInput。
            if (version == 1) {
                std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
                std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
                LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
            } else if (version == 2) {
                std::string inputRMSWeightName =
                        "transformer.encoder.layers." + std::to_string(i) + ".input_layernorm.weight";
                RMSNorm(hiddenStates, weight[inputRMSWeightName], 1e-5, attenInput);
            }

            // 使用 Linear 函数对 attenInput 进行线性变换，得到 qkv。
            std::string qkvWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.weight";
            std::string qkvBiasName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.bias";
            if (!adapterName.empty()) {
                std::string peftType = weight.peftDict[adapterName]["peft_type"];
                if (peftType == "LORA") {
                    std::string loraAWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.lora_A." + adapterName + ".weight";
                    std::string loraBWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.lora_B." + adapterName + ".weight";
                    LoraLayer(attenInput, weight[qkvWeightName], weight[loraAWeightName], weight[loraBWeightName], weight[qkvBiasName], qkv, weight.peftDict[adapterName]);
                } else if (peftType == "IA3") {
                    std::string ia3WeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.ia3_l" + adapterName + ".weight";
                    IA3Layer(attenInput, weight[qkvWeightName], weight[ia3WeightName], weight[qkvBiasName], qkv, weight.peftDict[adapterName]);
                }
            } else {
                Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);
            }

            // 对 qkv 进行一些操作（如 Reshape、Split、RotatePosition2D
            // 或 NearlyRotatePosition2D），得到 Q、K、V。
            if (version == 1) {
                qkv.Reshape({qkv.dims[0], qkv.dims[1], num_attention_heads, -1});
                int per = qkv.dims.back() / 3;
                Split(qkv, -1, 0, per, q);
                Split(qkv, -1, per, per * 2, k);
                Split(qkv, -1, per * 2, per * 3, v);
                fastllm::RotatePosition2D(q, positionIds, sinData, cosData, rotary_dim);
                fastllm::RotatePosition2D(k, positionIds, sinData, cosData, rotary_dim);
            } else if (version == 2) {
                int qLen = embed_dim, kvLen = (qkv.dims.back() - embed_dim) / 2;
                Split(qkv, -1, 0, qLen, q);
                Split(qkv, -1, qLen, qLen + kvLen, k);
                Split(qkv, -1, qLen + kvLen, qLen + kvLen + kvLen, v);
                q.Reshape({q.dims[0], q.dims[1], -1, embed_dim / num_attention_heads});
                k.Reshape({k.dims[0], k.dims[1], -1, embed_dim / num_attention_heads});
                v.Reshape({v.dims[0], v.dims[1], -1, embed_dim / num_attention_heads});
                fastllm::NearlyRotatePosition2D(q, positionIds, sinData, cosData, rotary_dim);
                fastllm::NearlyRotatePosition2D(k, positionIds, sinData, cosData, rotary_dim);
            }

            // q, k, v, shape => 【seq_length, batch, num_attention_head, hidden_size / num_attention_head】

            // 从 pastKeyValues 中获取第 i 个元素，该元素是一个 pair，将其两个元素分别赋给
            // pastKey 和 pastValue，注意这里使用了引用，所以对 pastKey 和 pastValue
            // 的修改会影响 pastKeyValues。
            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;

            // 如果 GetKVCacheInCPU() 返回 true，则将 pastKey 和 pastValue 的 lockInCPU
            // 属性设置为 true，这可能意味着这两个数据将被锁定在 CPU 上；
            // 否则，将 pastKey 和 pastValue 移动到 CUDA 设备上，
            // 这可能意味着这两个数据将被移动到 GPU 上进行计算。
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(DataDevice::CUDA);
                pastValue.ToDevice(DataDevice::CUDA);
            };

            // 调整 K 和 V 的形状。
            k.Resize({k.dims[0], k.dims[1] * k.dims[2], k.dims[3]});
            v.Resize({v.dims[0], v.dims[1] * v.dims[2], v.dims[3]});

            // 对 K 和 V 进行置换，将batch维度和序列长度维度交换。
            // 【seq_length, batch * num_attention_head, hidden_size / num_attention_head】=>
            // 【batch * num_attention_head, seq_length, hidden_size / num_attention_head】
            PermuteSelf(k, {1, 0, 2});
            PermuteSelf(v, {1, 0, 2});

            // 定义一个变量 unitLen，并赋值为 64。#ifdef USE_CUDA 判断是否定义了
            // USE_CUDA，如果定义了，则将 unitLen 设置为 128。这个变量可能与后面的内存扩展有关。
            int unitLen = 64;
#ifdef USE_CUDA
            unitLen = 128;
#endif
            // 接下来的两个 while 循环对 pastKey 和 pastValue 进行扩展。
            // 具体来说，如果 pastKey 或 pastValue 的大小小于 K 或 V 的大小，
            // 则将 pastKey 或 pastValue 的大小扩展到满足需要的最小值。
            // 这里的 unitLen 可能是为了保证扩展后的大小是 unitLen 的整数倍，以提高内存访问效率。
            // 这里扩展的原因是因为KV Cache会把pastKey，pastValue越concat越大，所以需要动态扩容
            while ((pastKey.dims.size() == 0 &&
                    (pastKey.expansionDims.size() == 0 || k.dims[1] > pastKey.expansionDims[1]))
                   || (pastKey.dims.size() > 0 && (pastKey.expansionDims.size() == 0 ||
                                                   pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1]))) {
                std::vector<int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector<int>{k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                    if (generationConfig.output_token_limit > 0) {
                        newDims[1] = k.dims[1] + generationConfig.output_token_limit;
                    }
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }

            while ((pastValue.dims.size() == 0 &&
                    (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                   || (pastValue.dims.size() > 0 && (pastValue.expansionDims.size() == 0 ||
                                                     pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1]))) {
                std::vector<int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector<int>{v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                    if (generationConfig.output_token_limit > 0) {
                        newDims[1] = k.dims[1] + generationConfig.output_token_limit;
                    }
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }

            // KV Cache的concat过程
            CatDirect(pastKey, k, 1);
            CatDirect(pastValue, v, 1);

            //q.shape 【seq_length, batch, num_attention_head, hidden_size / num_attention_head】
            // outputSize 【batch, num_attention_head, query seq_length, pastKey seq_length】
            std::vector<int> outputSize = {q.dims[1], q.dims[2], q.dims[0], pastKey.dims[1]};

            // q.shape 【seq_length, batch * num_attention_head, hidden_size / num_attention_head】
            q.Reshape({q.dims[0], q.dims[1] * q.dims[2], q.dims[3]});
            // q.shape 【batch * num_attention_head, seq_length, hidden_size / num_attention_head】
            PermuteSelf(q, {1, 0, 2});
            //Attention(q, pastKey, pastValue, attentionMask, contextLayer, q.dims[0] / pastKey.dims[0], 1.0 / scale_attn, 1);


            // 1.2 Attention
            // 1.2.0 q * k^T
            // q.shape 【batch * num_attention_head， query seq_length， hidden_size / num_attention_head】
            q.Reshape({pastKey.dims[0], -1, q.dims[2]});

            // pastKey.shape 【batch * num_attention_head， pastKey seq_length，hidden_size / num_attention_head】
            // pastKey^T.shape 【batch * num_attention_head，hidden_size / num_attention_head，pastKey seq_length】
            // attnProbs.shape 【batch * num_attention_head，query seq_length, pastKey seq_length】
            MatMulTransB(q, pastKey, attnProbs, 1.0 / (scale_attn * (i + 1)));
            attnProbs.Reshape(outputSize);

            // 1.2.1 Mask
            // 如果 attentionMask 的维度不为0，那么就对注意力概率（attnProbs）应用 attentionMask，
            // 这是注意力机制中的一个重要步骤，可以屏蔽掉一些不需要的信息。
            if (attentionMask.dims.size() != 0) {
                AttentionMask(attnProbs, attentionMask, -10000);
            }

            // 1.2.2 softmax
            // 将注意力概率与i + 1相乘，然后对结果应用 softmax 函数，使得所有的注意力概率之和为 1。
            Mul(attnProbs, i + 1, attnProbs);
            Softmax(attnProbs, attnProbs, -1);

            // 定义输出的大小，然后根据这个大小重新reshape attnProbs。
            // outputSize.shape [1, batch * num_attention_head, query seq_length, pastKey seq_length]
            outputSize = {1, pastValue.dims[0], q.dims[1], pastValue.dims[1]};
            attnProbs.Reshape({outputSize[0] * outputSize[1], outputSize[2], -1});
            // 1.2.3 prob * v

            attnProbs.Reshape({pastValue.dims[0], -1, attnProbs.dims[2]});
            MatMul(attnProbs, pastValue, contextLayer);
            contextLayer.Reshape({batch, num_attention_heads, maxLen, -1});
            PermuteSelf(contextLayer, {2, 0, 1, 3});
            contextLayer.Reshape({contextLayer.dims[0], contextLayer.dims[1], embed_dim});

            // 1.2.4 dense
            std::string denseWeightName = weightPre + std::to_string(i) + weightMiddle + ".dense.weight";
            std::string denseBiasName = weightPre + std::to_string(i) + weightMiddle + ".dense.bias";
            Linear(contextLayer, weight[denseWeightName], weight[denseBiasName], attnOutput);

            // 1.3
            if (GetVersion() == 1) {
                float alpha = sqrt(2 * block_cnt);
                Mul(attenInput, alpha, hiddenStates);
                AddTo(hiddenStates, attnOutput);
                std::string postLNWeightName =
                        "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                std::string postLNBiasName =
                        "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.bias";
                LayerNorm(hiddenStates, weight[postLNWeightName], weight[postLNBiasName], -1, mlpInput);
                // 1.4 MLP
                std::string fcInKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
                std::string fcOutKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
                Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
                GeluNew(middle, middle);
                Linear(middle, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
                AddTo(hiddenStates, mlpInput, alpha);
            } else {
                AddTo(hiddenStates, attnOutput);
                std::string postRMSWeightName =
                        "transformer.encoder.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                Mul(hiddenStates, 1.0, temp);
                RMSNorm(hiddenStates, weight[postRMSWeightName], 1e-5, mlpInput);
                // 1.4 MLP
                std::string fcInKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
                std::string fcOutKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
                Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
                Swiglu(middle, middle2);
                Linear(middle2, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
                AddTo(hiddenStates, temp);
            }
        }

        // 定义 logits 和 topk，这可能是为了保存模型的输出和最终的预测结果。
        Data logits, topk;
        // 接下来的代码块根据 version 的值进行不同的操作，主要包括应用层归一化和线性变换，
        // 得到模型的输出 logits。
        if (version == 1) {
            LayerNorm(hiddenStates, weight["transformer.final_layernorm.weight"],
                      weight["transformer.final_layernorm.bias"], -1, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        } else {
            RMSNorm(hiddenStates, weight["transformer.encoder.final_layernorm.weight"], 1e-5, hiddenStates);
            Linear(hiddenStates, weight["transformer.output_layer.weight"], Data(), logits);
        }
        if (generationConfig.output_logits && retLogits != nullptr) {
            int size = logits.dims.back();
            logits.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b;
                (*retLogits)[b]->resize(size);
                memcpy((float*)(*retLogits)[b]->data(), ((float*)logits.cpuData) + base * size, size * logits.unitSize);
            }
        }

        // 如果生成配置指定了简单的贪心策略，那么就从 logits 中找出最大的值作为预测结果；
        // 否则，使用 LLMSampling 函数根据 logits 和生成配置来选择预测结果。
        if (generationConfig.IsSimpleGreedy()) {
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b;
                lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
            }
        } else if (!lastTokens.units.empty()) {
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b;
                lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
            }
        }
        return lastRet;
    }
    //还有一个同名的ForwardBatch函数，
    //和上面这个函数的区别在于它支持对不同的seq_length组的batch进行推理，简单来说就是在上面的基础上对batch进行了一个loop。

    std::vector <int> ChatGLMModel::ForwardBatch(
            int batch,
            const Data &inputIds,
            const std::vector <Data*> &attentionMask,
            const std::vector <Data*> &positionIds,
            const std::vector <int> &seqLens,
            std::vector <std::pair <Data*, Data*> > &pastKeyValues,
            const std::vector <GenerationConfig> &generationConfigs,
            const LastTokensManager &lastTokens,
            std::vector <std::vector <float>*> *retLogits) {
        if (this->weight.dicts.find("rope_ratio") != this->weight.dicts.end()) {
            UpdateSinCos(atof(this->weight.dicts["rope_ratio"].c_str()));
        }
        int seqLen = inputIds.dims[1];
        sinData.ToDevice(DataDevice::CUDA);
        cosData.ToDevice(DataDevice::CUDA);
        int version = GetVersion();
        std::string weightPre, weightMiddle;
        if (version == 1) {
            weightPre = "transformer.layers.";
            weightMiddle = ".attention";
        } else if (version == 2) {
            weightPre = "transformer.encoder.layers.";
            weightMiddle = ".self_attention";
        }
        Data inputEmbeddings;
        Data inputIdsPermute;
        Permute(inputIds, {1, 0}, inputIdsPermute);
        Embedding(inputIdsPermute, this->weight["transformer" + std::string((version == 2 ? ".embedding" : "")) +
                                                ".word_embeddings.weight"], inputEmbeddings);
        Data &hiddenStates = inputEmbeddings;
        hiddenStates.ToDevice(DataDevice::CUDA);
        Data attenInput;
        Data qkv, q, k, v;
        Data attnOutput;
        Data mlpInput, middle, middle2;
        std::vector <Data> attnProbs;
        std::vector <Data> curContextLayer;
        std::vector <Data> curKs, curVs, curQs;
        attnProbs.resize(batch);
        curContextLayer.resize(batch);
        curKs.resize(batch);
        curVs.resize(batch);
        curQs.resize(batch);
        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }

        if (batch > 1) {
            positionIds[0]->Expansion({2, seqLen});
            for (int i = 1; i < batch; i++) {
                CatDirect(*(Data*)positionIds[0], *(Data*)positionIds[i], 1);
            }
        }
        std::vector <Data*> keys, values, qs, attns, masks, contexts;
        keys.resize(batch);
        values.resize(batch);
        qs.resize(batch);
        attns.resize(batch);
        masks.resize(batch);
        contexts.resize(batch);

        std::vector <Data*> pointersK, pointersV, pointersQ;
        pointersK.resize(batch);
        pointersV.resize(batch);
        pointersQ.resize(batch);

        std::vector <std::vector <int> > outputSizes;
        outputSizes.resize(batch);
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            if (version == 1) {
                std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
                std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
                LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
            } else if (version == 2) {
                std::string inputRMSWeightName =
                        "transformer.encoder.layers." + std::to_string(i) + ".input_layernorm.weight";
                RMSNorm(hiddenStates, weight[inputRMSWeightName], 1e-5, attenInput);
            }

            std::string qkvWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.weight";
            std::string qkvBiasName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.bias";
            if (!adapterName.empty()) {
                std::string peftType = weight.peftDict[adapterName]["peft_type"];
                if (peftType == "LORA") {
                    std::string loraAWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.lora_A." + adapterName + ".weight";
                    std::string loraBWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.lora_B." + adapterName + ".weight";
                    LoraLayer(attenInput, weight[qkvWeightName], weight[loraAWeightName], weight[loraBWeightName], weight[qkvBiasName], qkv, weight.peftDict[adapterName]);
                } else if (peftType == "IA3") {
                    std::string ia3WeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.ia3_l" + adapterName + ".weight";
                    IA3Layer(attenInput, weight[qkvWeightName], weight[ia3WeightName], weight[qkvBiasName], qkv, weight.peftDict[adapterName]);
                }
            } else {
                Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);
            }

            if (version == 1) {
                qkv.Reshape({qkv.dims[0], qkv.dims[1], num_attention_heads, -1});
                int per = qkv.dims.back() / 3;
                Split(qkv, -1, 0, per, q);
                Split(qkv, -1, per, per * 2, k);
                Split(qkv, -1, per * 2, per * 3, v);
            } else if (version == 2) {
                int qLen = embed_dim, kvLen = (qkv.dims.back() - embed_dim) / 2;
                Split(qkv, -1, 0, qLen, q);
                Split(qkv, -1, qLen, qLen + kvLen, k);
                Split(qkv, -1, qLen + kvLen, qLen + kvLen + kvLen, v);
                q.Reshape({q.dims[0], q.dims[1], -1, embed_dim / num_attention_heads});
                k.Reshape({k.dims[0], k.dims[1], -1, embed_dim / num_attention_heads});
                v.Reshape({v.dims[0], v.dims[1], -1, embed_dim / num_attention_heads});
            }

            if (version == 1) {
                fastllm::RotatePosition2D(q, *positionIds[0], sinData, cosData, rotary_dim);
                fastllm::RotatePosition2D(k, *positionIds[0], sinData, cosData, rotary_dim);
            } else if (version == 2) {
                fastllm::NearlyRotatePosition2D(q, *positionIds[0], sinData, cosData, rotary_dim);
                fastllm::NearlyRotatePosition2D(k, *positionIds[0], sinData, cosData, rotary_dim);
            }

            k.Resize({k.dims[0], k.dims[1] * k.dims[2], k.dims[3]});
            v.Resize({v.dims[0], v.dims[1] * v.dims[2], v.dims[3]});
            q.Resize({q.dims[0], q.dims[1] * q.dims[2], q.dims[3]});

            Data contextLayer = Data(DataType::FLOAT32);
            int total = 0;
            if (all1 && batch > 1) {
                for (int b = 0; b < batch; b++) {
                    pointersK[b] = (&curKs[b]);
                    pointersV[b] = (&curVs[b]);
                    pointersQ[b] = (&curQs[b]);
                }
                SplitBatch(k, 0, batch, pointersK);
                SplitBatch(v, 0, batch, pointersV);
                SplitBatch(q, 0, batch, pointersQ);
                total = batch;
                for (int b = 0; b < batch; b++) {
                    auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                    std::swap(k.dims[0], k.dims[1]);
                    k.strides[0] = k.dims[1] * k.dims[2]; k.strides[1] = k.dims[2];
                    std::swap(v.dims[0], v.dims[1]);
                    v.strides[0] = v.dims[1] * v.dims[2]; v.strides[1] = v.dims[2];
                    std::swap(q.dims[0], q.dims[1]);
                    q.strides[0] = q.dims[1] * q.dims[2]; q.strides[1] = q.dims[2];
                }
            } else {
                PermuteSelf(k, {1, 0, 2});
                PermuteSelf(v, {1, 0, 2});
                PermuteSelf(q, {1, 0, 2});
                for (int b = 0; b < batch; b++) {
                    Split(k, 1, total, total + seqLens[b], curKs[b]);
                    Split(v, 1, total, total + seqLens[b], curVs[b]);
                    Split(q, 1, total, total + seqLens[b], curQs[b]);
                    total += seqLens[b];
                }
            }

            for (int b = 0; b < batch; b++) {
                auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt +
                                                                                                     i].second;
                if (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] <= pastKey.expansionDims[1]) {
                    continue;
                }

                pastKey.ToDevice(DataDevice::CUDA);
                pastValue.ToDevice(DataDevice::CUDA);

                int unitLen = 64;
#ifdef USE_CUDA
                unitLen = 128;
#endif
                while ((pastKey.dims.size() == 0 &&
                        (pastKey.expansionDims.size() == 0 || k.dims[1] > pastKey.expansionDims[1]))
                       || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector<int>{k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                        if (generationConfigs[b].output_token_limit > 0) {
                            newDims[1] = std::min(newDims[1], k.dims[1] + generationConfigs[b].output_token_limit);
                        }
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }

                while ((pastValue.dims.size() == 0 &&
                        (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                       || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector<int>{v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                        if (generationConfigs[b].output_token_limit > 0) {
                            newDims[1] = std::min(newDims[1], k.dims[1] + generationConfigs[b].output_token_limit);
                        }
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }
            }
            for (int b = 0; b < batch; b++) {
                keys[b] = (pastKeyValues[b * block_cnt + i].first);
                values[b] = (pastKeyValues[b * block_cnt + i].second);
                pointersK[b] = (&curKs[b]);
                pointersV[b] = (&curVs[b]);
            }
            CatDirectBatch(keys, pointersK, 1);
            CatDirectBatch(values, pointersV, 1);
            if (all1 && batch > 1) {
                for (int b = 0; b < batch; b++) {
                    qs[b] = (&curQs[b]);
                    keys[b] = (pastKeyValues[b * block_cnt + i].first);
                    values[b] = (pastKeyValues[b * block_cnt + i].second);
                    masks[b] = attentionMask[b];
                    contexts[b] = (&curContextLayer[b]);

                    outputSizes[b] = {1, qs[b]->dims[0], qs[b]->dims[1], keys[b]->dims[1]};
                }
                AttentionBatch(qs, keys, values, masks, contexts, qs[0]->dims[0] / values[0]->dims[0], 1.0 / scale_attn, 1);
            } else {
                for (int b = 0; b < batch; b++) {
                    auto &q = curQs[b];
                    Data &pastKey = *pastKeyValues[b * block_cnt + i].first;
                    outputSizes[b] = {1, q.dims[0], q.dims[1], pastKey.dims[1]};
                    q.Reshape({pastKey.dims[0], -1, q.dims[2]});
                }

                // 1.2 Attention
                // 1.2.0 q * k^T
                if (all1 && batch > 1) {
                    for (int b = 0; b < batch; b++) {
                        qs[b] = (&curQs[b]);
                        keys[b] = (pastKeyValues[b * block_cnt + i].first);
                        attns[b] = (&attnProbs[b]);
                    }
                    MatMulTransBBatch(qs, keys, attns, 1.0 / (scale_attn * (i + 1)));
                } else {
                    for (int b = 0; b < batch; b++) {
                        auto &q = curQs[b];
                        Data &pastKey = *pastKeyValues[b * block_cnt + i].first;
                        MatMulTransB(q, pastKey, attnProbs[b], 1.0 / (scale_attn * (i + 1)));
                    }
                }

                for (int b = 0; b < batch; b++) {
                    attnProbs[b].Reshape(outputSizes[b]);
                    // 1.2.1 Mask
                    if (attentionMask[b] != nullptr) {
                        AttentionMask(attnProbs[b], *attentionMask[b], -10000);
                    }
                }

                // 1.2.2 softmax
                for (int i = 0; i < attnProbs.size(); i++) {
                    attns[i] = (&attnProbs[i]);
                }
                MulBatch(attns, i + 1, attns);
                SoftmaxBatch(attns, attns, -1);

                for (int b = 0; b < batch; b++) {
                    Data &pastValue = *pastKeyValues[b * block_cnt + i].second;
                    outputSizes[b] = {1, num_attention_heads, -1, pastValue.dims[2]};
                    attnProbs[b].Reshape({pastValue.dims[0], -1, attnProbs[b].dims[3]});
                }

                // 1.2.3 prob * v
                if (all1 && batch > 1) {
                    for (int b = 0; b < batch; b++) {
                        attns[b] = (&attnProbs[b]);
                        values[b] = (pastKeyValues[b * block_cnt + i].second);
                        contexts[b] = (&curContextLayer[b]);
                    }
                    MatMulBatch(attns, values, contexts);
                } else {
                    for (int b = 0; b < batch; b++) {
                        Data &pastValue = *pastKeyValues[b * block_cnt + i].second;
                        MatMul(attnProbs[b], pastValue, curContextLayer[b]);
                    }
                }
            }
            if (all1) {
                for (int b = 0; b < batch; b++) {
                    curContextLayer[b].dims[0] = outputSizes[b][2];
                    curContextLayer[b].dims[1] = outputSizes[b][0];
                    curContextLayer[b].dims[2] = embed_dim;
                    curContextLayer[b].strides[0] = curContextLayer[b].dims[1] * curContextLayer[b].dims[2];
                    curContextLayer[b].strides[1] = curContextLayer[b].dims[2];
                    curContextLayer[b].strides[2] = 1;
                }
            } else {
                for (int b = 0; b < batch; b++) {
                    curContextLayer[b].Reshape(outputSizes[b]);
                    PermuteSelf(curContextLayer[b], {2, 0, 1, 3});
                    curContextLayer[b].Reshape({curContextLayer[b].dims[0], curContextLayer[b].dims[1], embed_dim});
                }
            }

            if (all1 && batch > 1) {
                for (int b = 0; b < batch; b++) {
                    contexts[b] = (&curContextLayer[b]);
                }
                CatBatch(contexts, 0, contextLayer);
            } else {
                for (int b = 0; b < batch; b++) {
                    if (contextLayer.dims.size() == 0) {
                        std::vector<int> dims = curContextLayer[b].dims;
                        dims[0] = total;
                        contextLayer.Expansion(dims);
                    }
                    contextLayer.ToDevice(DataDevice::CUDA);
                    CatDirect(contextLayer, curContextLayer[b], 0);
                }
            }
            // 1.2.4 dense
            std::string denseWeightName = weightPre + std::to_string(i) + weightMiddle + ".dense.weight";
            std::string denseBiasName = weightPre + std::to_string(i) + weightMiddle + ".dense.bias";
            Linear(contextLayer, weight[denseWeightName], weight[denseBiasName], attnOutput);
            if (GetVersion() == 1) {
                float alpha = sqrt(2 * block_cnt);
                Mul(attenInput, alpha, hiddenStates);
                AddTo(hiddenStates, attnOutput);
                std::string postLNWeightName =
                        "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                std::string postLNBiasName =
                        "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.bias";
                LayerNorm(hiddenStates, weight[postLNWeightName], weight[postLNBiasName], -1, mlpInput);
                // 1.4 MLP
                std::string fcInKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
                std::string fcOutKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
                Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
                GeluNew(middle, middle);
                Linear(middle, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
                AddTo(hiddenStates, mlpInput, alpha);
            } else {
                AddTo(hiddenStates, attnOutput);
                std::string postRMSWeightName =
                        "transformer.encoder.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                Data temp;
                Mul(hiddenStates, 1.0, temp);
                RMSNorm(hiddenStates, weight[postRMSWeightName], 1e-5, mlpInput);
                // 1.4 MLP
                std::string fcInKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
                std::string fcOutKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
                Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
                Swiglu(middle, middle2);
                Linear(middle2, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
                AddTo(hiddenStates, temp);
            }
        }
        Data logits;
        if (version == 1) {
            LayerNorm(hiddenStates, weight["transformer.final_layernorm.weight"],
                      weight["transformer.final_layernorm.bias"], -1, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        } else {
            RMSNorm(hiddenStates, weight["transformer.encoder.final_layernorm.weight"], 1e-5, hiddenStates);
            Linear(hiddenStates, weight["transformer.output_layer.weight"], Data(), logits);
        }
        std::vector <int> lastRet;
        int total = 0;
        Data curLogit;
        for (int b = 0; b < batch; b++) {
            Split(logits, 0, total + seqLens[b] - 1, total + seqLens[b], curLogit);
            if (generationConfigs[b].output_logits && retLogits != nullptr && (*retLogits)[b] != nullptr) {
                curLogit.ToDevice(DataDevice::CPU);
                (*retLogits)[b]->resize(curLogit.Count(0));
                memcpy((float*)(*retLogits)[b]->data(), (float*)curLogit.cpuData, curLogit.GetBytes());
            }
            if (generationConfigs[b].IsSimpleGreedy()) {
                Data topk;
                TopK(curLogit, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                lastRet.push_back((int) (((float *) topk.cpuData)[0] + 1e-3));
            } else {
                lastRet.push_back(LLMSampling(curLogit, 0, generationConfigs[b], lastTokens.units[b]));
            }
            total += seqLens[b];
        }
        return lastRet;
    }

    void ChatGLMModel::FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                                     const std::map <std::string, int> &params,
                                     Data &inputIds, Data &attentionMask, Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);

        // 将 attentionMask 和 positionIds 移动到 CPU 设备上。
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        // 在模型的权重字典中查找“gmask_token_id”，如果找到了就将其值转化为整数，
        // 如果没找到就将其设为130001。
        int gmask_token_id = this->weight.dicts.find("gmask_token_id") != this->weight.dicts.end() ?
                             atoi(this->weight.dicts["gmask_token_id"].c_str()) : 130001;
        int index = params.find("index")->second;
        int promptLen = params.find("promptLen")->second;

        if (index == 0) {
            for (auto &ids: inputTokens) {
                // 根据版本号，在 ids 的末尾或开头插入特定的整数值。
                if (GetVersion() == 1) {
                    ids.push_back(gmask_token_id);
                    ids.push_back(bos_token_id);
                } else if (GetVersion() == 2) {
                    if (ids.size() < 2 || ids[0] != 64790 || ids[1] != 64792) {
                        ids.insert(ids.begin(), 64792);
                        ids.insert(ids.begin(), 64790);
                    }
                }
            }

            int seqLen = inputTokens[0].size();
            std::vector<float> vmask = std::vector<float>(seqLen * seqLen, 0);
            std::vector<float> vpids = std::vector<float>(seqLen * 2, 0);
            for (int i = 0; i < seqLen - 1; i++) {
                vmask[i * seqLen + seqLen - 1] = 1;
                vpids[i] = i;
            }

            // 为 vmask 和 vpids 初始化值。
            vpids[seqLen - 1] = seqLen - 2;
            vpids[seqLen * 2 - 1] = 1;

            // 如果版本号为 2，那么重新为 vmask 和 vpids 分配值。
            if (GetVersion() == 2) {
                for (int i = 0; i < seqLen; i++) {
                    vpids[i] = i;
                    for (int j = i + 1; j < seqLen; j++) {
                        vmask[i * seqLen + j] = 1;
                    }
                }
            }

            // 根据 ids 创建一个新的 Data 对象，并将其复制给 inputIds。
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, inputTokens[0]));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {seqLen, seqLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {2, seqLen}, vpids));
        } else {
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, inputTokens[0]));

            // 更新 attentionMask 和 positionIds。
            attentionMask = Data();
            if (GetVersion() == 1) {
                positionIds.CopyFrom(Data(DataType::FLOAT32, {2, 1}, {(float) promptLen, (float) (index + 1)}));
            } else {
                positionIds.CopyFrom(Data(DataType::FLOAT32, {2, 1}, {(float) promptLen + index + 1, (float) (index + 1)}));
            }
        }
    }

    void ChatGLMModel::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            // 在模型的权重字典中查找“gmask_token_id”，如果找到了就将其值转化为整数，
            // 如果没找到就将其设为130001。
            int gmask_token_id = this->weight.dicts.find("gmask_token_id") != this->weight.dicts.end() ?
                                 atoi(this->weight.dicts["gmask_token_id"].c_str()) : 130001;
            std::vector<int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int) inputTokens[i].size() + 2);
                seqLens[i] = (int) inputTokens[i].size();
            }

            std::vector<float> ids = std::vector<float>(batch * maxLen, 0);
            std::vector<float> vpids = std::vector<float>(batch * 2 * maxLen, 0);
            std::vector<float> vmask = std::vector<float>(batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                if (GetVersion() == 1) {
                    auto &tokens = inputTokens[i];
                    int len = tokens.size(), base = maxLen - 2 - len;
                    for (int j = 0; j < len; j++) {
                        ids[i * maxLen + base + j] = tokens[j];
                    }
                    ids[i * maxLen + base + len] = gmask_token_id;
                    ids[i * maxLen + base + len + 1] = bos_token_id;
                    len += 2;
                    for (int j = 0; j < len - 1; j++) {
                        vpids[i * 2 * maxLen + base + j] = j;
                    }
                    vpids[i * 2 * maxLen + base + len - 1] = len - 2;
                    vpids[i * 2 * maxLen + maxLen + base + len - 1] = 1;
                    std::fill(vmask.data() + i * maxLen * maxLen,
                              vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                    for (int j = maxLen - len; j < maxLen; j++) {
                        std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                                  vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                    }
                    for (int j = 0; j < len - 1; j++) {
                        vmask[i * maxLen * maxLen + (base + j) * maxLen + base + len - 1] = 1;
                    }
                } else {
                    auto &tokens = inputTokens[i];
                    int len = tokens.size(), base = maxLen - 2 - len;
                    ids[i * maxLen + base] = 64790;
                    ids[i * maxLen + base + 1] = 64792;
                    for (int j = 0; j < len; j++) {
                        ids[i * maxLen + base + 2 + j] = tokens[j];
                    }
                    len += 2;
                    for (int j = 0; j < len; j++) {
                        vpids[i * 2 * maxLen + base + j] = j;
                    }

                    std::fill(vmask.data() + i * maxLen * maxLen,
                              vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                    for (int j = maxLen - len; j < maxLen; j++) {
                        std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                                  vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                    }
                    for (int j = 0; j < len; j++) {
                        for (int k = j + 1; k < len; k++) {
                            vmask[i * maxLen * maxLen + (base + j) * maxLen + base + k] = 1;
                        }
                    }
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch * 2, maxLen}, vpids));
        } else {
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            std::vector <float> pids = std::vector<float>(batch * 2);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen + 2, maxLen);
                pids[i * 2 + 1] = index + 1;
                if (GetVersion() == 1) {
                    pids[i * 2] = promptLen;
                } else {
                    pids[i * 2] = promptLen + index + 1;
                }
            }
            maxLen += index;
            std::vector<float> vmasks = std::vector<float>(batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                for (int j = 0; j < maxLen - index - promptLen - 2; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch * 2, 1}, pids));
        }
    }

    void ChatGLMModel::WarmUp() {
    	printf("Warmup...\n");
	    Data inputIds = Data(DataType::FLOAT32, {1, 1}, {(float)bos_token_id});

	    // 根据 vmask 和 vpids 创建 attentionMask 和 positionIds。
	    Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
	    Data positionIds = Data(DataType::FLOAT32, {2, 1}, {0, 0});

        // 创建一个包含 block_cnt 个空 Data 对象的向量 pastKeyValues。
	    std::vector <std::pair <Data, Data> > pastKeyValues;
	    for (int i = 0; i < block_cnt; i++) {
		    pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
		                                           Data(DataType::FLOAT32)));
	    }
	    Forward(inputIds, attentionMask, positionIds, pastKeyValues);
	    printf("finish.\n");
    }

    std::string ChatGLMModel::MakeInput(const std::string &history, int round, const std::string &input) {
        if (round == 0 && GetVersion() == 1) {
            return input;
        } else {
#if defined(_WIN32) or defined(_WIN64)
            std::vector <uint8_t> vask = {233, 151, 174, 239, 188, 154, 0};
            std::vector <uint8_t> vans = {231, 173, 148, 239, 188, 154, 0};
            std::string sask = (char*)vask.data();
            std::string sans = (char*)vans.data();
            return (history + ("[Round " + std::to_string(round) + "]\n\n" + sask + input + "\n\n" + sans));
#else
            return history + ("[Round " + std::to_string(round) + "]\n\n问：" + input + "\n\n答：");
#endif
        }
    }

    std::string ChatGLMModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
#if defined(_WIN32) or defined(_WIN64)
        std::vector <uint8_t> vask = {233, 151, 174, 239, 188, 154, 0};
        std::vector <uint8_t> vans = {231, 173, 148, 239, 188, 154, 0};
        std::string sask = (char*)vask.data();
        std::string sans = (char*)vans.data();
        return (history + ("[Round " + std::to_string(round) + "]\n\n" + sask + input + "\n\n" + sans + output + "\n"));
#else
        return (history + ("[Round " + std::to_string(round) + "]\n\n问：" + input + "\n\n答：" + output + "\n\n"));
#endif
    }

    int ChatGLMModel::GetVersion() {
        if (this->weight.weight.find("transformer.embedding.word_embeddings.weight") != this->weight.weight.end()) {
            return 2;
        } else {
            return 1;
        }
    }
}
