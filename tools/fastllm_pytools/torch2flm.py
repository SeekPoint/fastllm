'''
0x2. FastLLM chatglm-6b模型导出解析
首先解读一下FastLLM是如何导出huggingface的chatglm-6b模型的。

首先来看 fastllm/tools/fastllm_pytools/torch2flm.py 这个文件，这个文件实现了一个tofile函数用于将一个训练好的模型导出到一个文件中。
具体来说，它包括以下几个步骤：

    打开一个二进制文件，准备写入模型的数据。

    写入一个版本号，用于后续的兼容性检查。

    获取模型的配置信息，并将它们写入文件。如果提供了一些额外的配置参数，
    如 pre_prompt，user_role，bot_role，history_sep，也将它们添加到配置信息中。

    如果提供了分词器（tokenizer），将分词器的词汇表写入文件。
    如果分词器是一个句子片段模型（sentence piece model），那么还会写入一些额外的信息。

    获取模型的权重（包含在模型的状态字典中），并将它们写入文件。
    权重的名字和形状都会被写入文件，以便于后续正确地加载模型。

    在每写入一个权重后，打印进度信息，以便于用户知道当前的进度。

    最后，关闭文件。

更详细的解释可以请看：
'''
# struct 是Python的一个内置模块，提供了一些函数来解析打包的二进制数据。
# 在这个代码中，它被用于将整数和字符串转换为二进制格式。
import struct

import numpy as np
import torch

# 定义一个函数 writeString，它接受两个参数：一个文件对象 fo 和一个字符串 s。
def writeString(fo, s):
    # struct.pack 函数将 len(s)（字符串 s 的长度）打包为一个二进制字符串，
    # 然后 fo.write 将这个二进制字符串写入文件。
    fo.write(struct.pack('i', len(s)));
    # s.encode() 将字符串 s 转换为二进制格式，然后 fo.write 将这个二进制字符串写入文件。
    fo.write(s.encode());

# 定义一个函数 writeKeyValue，它接受三个参数：一个文件对象 fo，一个键 key 和一个值 value。
def writeKeyValue(fo, key, value):
    writeString(fo, key);
    writeString(fo, value);

fastllm_data_type_dict = {
    "int4": 8,
    "int8": 3,
    "float16": 7,
    "float32": 0,
}
fastllm_weight_type_dict = {
    "linear": 1,
    "embedding": 2
}

v = np.random.randint(-127, 127, [10, 20]);
temp = v;
c_max = np.expand_dims(np.abs(v).max(axis = -1), -1)
c_scale = c_max / 127.0
v = (v / c_scale + 128.5).clip(1, 255).astype(np.uint8)

def write_int8(fo, v):
    c_max = np.expand_dims(np.abs(v).max(axis = -1), -1).clip(0.1, 1e100)
    c_scale = c_max / 127.0
    v = (v / c_scale + 128.5).clip(1, 255).astype(np.uint8)
    fo.write(struct.pack('i', 3))
    fo.write(struct.pack('i', 0))
    for i in range(c_max.shape[0]):
        fo.write(struct.pack('f', -c_max[i][0]));
        fo.write(struct.pack('f', c_max[i][0]));
    fo.write(v.data)

def write_int4(fo, v):
    c_min = np.expand_dims(-np.abs(v).max(axis = -1), -1)
    c_max = np.expand_dims(np.abs(v).max(axis = -1), -1)
    c_scale = c_max / 7.0
    c_min = c_scale * -8.0
    v = (v - c_min) / c_scale
    v = (v + 0.5).astype(np.int8).clip(0, 15).astype(np.uint8)
    v = v[:, 0::2] * 16 + v[:, 1::2]
    fo.write(struct.pack('i', 8))
    fo.write(struct.pack('i', 0))
    for i in range(c_min.shape[0]):
        fo.write(struct.pack('f', c_min[i][0]));
        fo.write(struct.pack('f', c_max[i][0]));
    fo.write(v.data)

# 这段Python代码的主要作用是将模型的状态字典（state_dict）以及一些模型的配置信息保存到一个文件中。
# 定义了一个函数 tofile，它接受七个参数：一个文件路径 exportPath，一个模型对象 model，
# 和五个可选参数 tokenizer，pre_prompt，user_role，bot_role，history_sep。
def tofile(exportPath,
           model,
           tokenizer = None,
           pre_prompt = None,
           user_role = None,
           bot_role = None,
           history_sep = None,
           dtype = "float16"):
    if (dtype not in fastllm_data_type_dict):
        print("dtype should in ", list(fastllm_data_type_dict.keys()))
        exit(0)

    # 获取模型的状态字典。状态字典是一个Python字典，它保存了模型的所有权重和偏置。
    dict = model.state_dict();
    # 打开一个文件以写入二进制数据。
    fo = open(exportPath, "wb");

    # 0. version id
    # 写入一个版本号 2。
    fo.write(struct.pack('i', 2));

    # 0.1 model info
    modelInfo = model.config.__dict__ #  获取模型配置的字典。

    if model.generation_config is not None:
        modelInfo.update(model.generation_config.__dict__)
    if ("model_type" not in modelInfo):
        print("unknown model_type.")
        exit(0)

    # 如果提供了 pre_prompt，user_role，bot_role，history_sep，则将它们添加到 modelInfo 中
    if (pre_prompt):
        modelInfo["pre_prompt"] = pre_prompt
    if (user_role):
        modelInfo["user_role"] = user_role
    if (bot_role):
        modelInfo["bot_role"] = bot_role
    if (history_sep):
        modelInfo["history_sep"] = history_sep

    # 如果模型是 "baichuan" 类型，并且模型有 "get_alibi_mask" 属性，
    # 则将一些额外的信息添加到 modelInfo 中。
    if (modelInfo["model_type"] == "baichuan" and hasattr(model, "model") and hasattr(model.model, "get_alibi_mask")):
        # Baichuan 2代
        modelInfo["use_alibi"] = "1"
        modelInfo["pre_prompt"] = ""
        modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.user_token_id) + "> ") if hasattr(model.generation_config, "user_token_id") else "";
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.assistant_token_id) + ">") if hasattr(model.generation_config, "assistant_token_id") else "";
        modelInfo["history_sep"] = ""
    if modelInfo["model_type"] == "qwen":
        if modelInfo["chat_format"] == "chatml":
            modelInfo["im_end_id"] = tokenizer.im_end_id
            modelInfo["im_start_id"] = tokenizer.im_start_id

    modelInfo["tokenizer_use_score"] = "1" # 分词带分数

    if hasattr(model, "peft_config"):
        adapter_size = len(model.peft_config)
        modelInfo["peft_size"] = adapter_size

    #  写入 modelInfo 的长度。
    fo.write(struct.pack('i', len(modelInfo)));

    # 遍历 modelInfo 的每一个键值对，并使用 writeKeyValue 函数将它们写入文件。
    for it in modelInfo.keys():
        writeKeyValue(fo, str(it), str(modelInfo[it]))

    if hasattr(model, "peft_config"):
        for adapter_name in model.peft_config.keys():
            adapter_dict = model.peft_config[adapter_name].__dict__
            writeString(fo, adapter_name)
            fo.write(struct.pack('i', len(adapter_dict)))
            for it in adapter_dict.keys():
                writeKeyValue(fo, str(it), str(adapter_dict[it]))

    # 1. vocab
    # 判断是否提供了分词器 tokenizer。分词器是一个将文本分解为词或其他有意义的符号的工具。
    if (tokenizer):
        if (hasattr(tokenizer, "tokenizer")):
            if (modelInfo['model_type'] == "qwen"):
                pass
            else:
                tokenizer = tokenizer.tokenizer
                
        # 如果分词器有 "sp_model" 属性，这意味着分词器是
        # 一个句子片段模型（sentence piece model），这是一种特殊的分词方法。
        if (hasattr(tokenizer, "sp_model")):
            # 获取句子片段模型的大小（即词汇表的大小）。
            piece_size = tokenizer.sp_model.piece_size()
            fo.write(struct.pack('i', piece_size))
            # for i in range(piece_size): 遍历词汇表中的每一个词。
            for i in range(piece_size):
                # 将词的ID转换为词本身，并将其编码为二进制字符串。
                s = tokenizer.sp_model.id_to_piece(i).encode()
                # 写入词的长度。
                fo.write(struct.pack('i', len(s)))
                # 遍历词的每一个字符，并将其写入文件。
                for c in s:
                    fo.write(struct.pack('i', c))
                #  写入词的ID。
                fo.write(struct.pack('i', i))
                fo.write(struct.pack('f', float(tokenizer.sp_model.get_score(i))))
        else:
            # 如果分词器没有 "sp_model" 属性，那么它就是一个普通的分词器。
            # 在这种情况下，它将获取词汇表，然后遍历词汇表中的每一个词，将词和对应的ID写入文件。
            vocab = tokenizer.get_vocab()
            fo.write(struct.pack('i', len(vocab)))
            for v in vocab.keys():
                if (modelInfo['model_type'] == "qwen"):
                    s = v
                else:
                    s = v.encode()
                if (modelInfo["model_type"] == "moss"):
                    s = [(ord(c) if c not in tokenizer.byte_decoder else tokenizer.byte_decoder[c]) for c in v]
                fo.write(struct.pack('i', len(s)))
                for c in s:
                    fo.write(struct.pack('i', c))
                fo.write(struct.pack('i', vocab[v]))
                fo.write(struct.pack('f', 1.0))
    else:
        # 如果没有提供分词器，那么它将写入一个0，表示词汇表的大小为0。
        fo.write(struct.pack('i', 0))

    weight_type_dict = {}
    module_dict = {}
    for key, m in model.named_modules():
        if (isinstance(m, torch.nn.Linear)):
            weight_type_dict[key + ".weight"] = "linear"
            module_dict[key + ".weight"] = m
        if (isinstance(m, torch.nn.Embedding)):
            weight_type_dict[key] = "embedding"

    # 2. weight
    # 写入模型状态字典的长度，即模型的权重数量。
    fo.write(struct.pack('i', len(dict)));
    tot = 0;

    # 遍历模型状态字典中的每一个键值对。键通常是权重的名字，值是权重的值
    for key in dict:
        ori_data_type = 0
        ori_np_data_type = np.float32
        cur_weight_type = 0
        if (key in weight_type_dict and weight_type_dict[key] in fastllm_weight_type_dict):
            cur_weight_type = fastllm_weight_type_dict[weight_type_dict[key]]
        to_data_type = 0
        if (cur_weight_type == 1):
            to_data_type = fastllm_data_type_dict[dtype]
            if (to_data_type == 7):
                ori_data_type = 7
                ori_np_data_type = np.float16
				
        # 将权重的值转换为NumPy数组，并确保其数据类型为float32。
        cur = dict[key].numpy().astype(ori_np_data_type)
        
        if hasattr(model, "peft_config"):
            weight_name = key.replace('base_model.model.', '')
            fo.write(struct.pack('i', len(weight_name)))
            fo.write(weight_name.encode())
        else:
            #  写入权重名字的长度。
            fo.write(struct.pack('i', len(key)))
            # 将权重名字编码为二进制字符串，然后写入文件。
            fo.write(key.encode())

        # 写入权重的维度数量。
        fo.write(struct.pack('i', len(cur.shape)))

        # 遍历权重的每一个维度，将其写入文件。
        for i in cur.shape:
            fo.write(struct.pack('i', i))

        if (to_data_type == 3):
            write_int8(fo, cur)
        elif (to_data_type == 8):
            write_int4(fo, cur)
        else:
            fo.write(struct.pack('i', to_data_type))
            # 将权重的值写入文件。
            fo.write(cur.data)
        # 记录已经写入的权重数量。
        tot += 1
        # 打印进度信息。
        print("output (", tot, "/", len(dict), end = " )\r")
    print("\nfinish.")
    fo.close() # 最后，关闭文件。