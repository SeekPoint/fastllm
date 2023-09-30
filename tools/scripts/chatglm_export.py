# 这段代码的主要功能是从预训练模型库中加载一个模型和对应的分词器，
# 并将它们导出为一个特定的文件格式（在这个例子中是 .flm 格式）。
# 以下是代码的详细解析：

# 导入Python的sys模块，它提供了一些与Python解释器和环境交互的函数和变量。
# 在这段代码中，它被用于获取命令行参数。
import sys

# 从transformers库中导入AutoTokenizer和AutoModel。transformers库是一个提供大量预训练模型的库，
# AutoTokenizer和AutoModel是用于自动加载这些预训练模型的工具。
from transformers import AutoTokenizer, AutoModel

# 从fastllm_pytools库中导入torch2flm模块。
# 这个模块可能包含了一些将PyTorch模型转换为.flm格式的函数。
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    # 从预训练模型库中加载一个分词器。"THUDM/chatglm-6b"是模型的名字。
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

    # 从预训练模型库中加载一个模型，并将它转换为浮点类型。
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

    # 将模型设置为评估模式。这是一个常见的操作，用于关闭模型的某些特性，
    # 如dropout和batch normalization。
    model = model.eval()

    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"

    # 获取命令行参数作为导出文件的路径。如果没有提供命令行参数，
    # 那么默认的文件名是"chatglm-6b-fp32.flm"。
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "chatglm-6b-' + dtype + '.flm"

    # 使用torch2flm的tofile函数将模型和分词器导出为.flm文件。
    torch2flm.tofile(exportPath, model, tokenizer, dtype = dtype)
