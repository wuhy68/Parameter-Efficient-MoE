# Parameter-Efficient Sparsity Crafting From Dense to Mixture-of-Experts for Instruction Tuning on General Tasks (EMNLP'24)

<a href="https://github.com/wuhy68/Parameter-Efficient-MoE/blob/master/LICENSE">
  <img src="https://img.shields.io/badge/Code_License-Apache_2.0-lightblue">
</a>
<a href="https://huggingface.co/hywu">
  <img src="https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg">
</a>

<img src="./img/Figure1.png">

## News
- 9/20/2024 - Our paper is accepted by EMNLP'24.
- 3/12/2024 - We release Qwen2idae-16x14B-v1.0 on 🤗 [HuggingFace](https://huggingface.co/hywu/Qwen2idae-16x14B-v1.0), which has strong performance in Math and Code with 15B activated params.
- 2/7/2024 - [Serp-ai](https://github.com/serp-ai/Parameter-Efficient-MoE) adds [unsloth](https://github.com/serp-ai/unsloth) support for faster and memory efficient training of our Parameter-Efficient Sparsity Crafting and releases new [sparsetral](https://huggingface.co/serpdotai/sparsetral-16x7B-v2) models based on mistral-7B.
- 1/10/2024 - Camelidae models are now available on 🤗 [HuggingFace](https://huggingface.co/hywu).
- 1/4/2024 - We release the paper, [Parameter-Efficient Sparsity Crafting From Dense to Mixture-of-Experts for Instruction Tuning on General Tasks](https://arxiv.org/abs/2401.02731).
- 12/22/2023 - We release the training [repo](https://github.com/wuhy68/Parameter-Efficient-MoE) that craft the dense model with LLaMA architecture to the MoE model.

## Introduction
We present Parameter-Efficient Sparsity Crafting to help dense models learn knowledge from different fields (including code and math). This appraoch perfrom instruction tuning and utilize MoE structure in an efficient way.

Parameter-Efficient Sparsity Crafting utilizes parameter efficient techiniques including [QLoRA](https://arxiv.org/abs/2305.14314) and [Adapter](https://arxiv.org/abs/1902.00751) to perfrom Efficient [Sparse Upcycling](https://arxiv.org/abs/2212.05055).

The repo supports the training of dense models ([LLaMA 2](https://arxiv.org/abs/2307.09288), [Yi](https://huggingface.co/01-ai), [Qwen1.5](https://github.com/QwenLM/Qwen1.5), etc.).

## Model Lists
| Camelidae Series | Download  
|---|---
Camelidae-8x7B   | 🤗 [HuggingFace](https://huggingface.co/hywu/Camelidae-8x7B)
Camelidae-8x13B  | 🤗 [HuggingFace](https://huggingface.co/hywu/Camelidae-8x13B)
Camelidae-8x34B  | 🤗 [HuggingFace](https://huggingface.co/hywu/Camelidae-8x34B) 

| Qwen2idae Series | Download  
|---|---
Qwen2idae-16x14B-v1.0   | 🤗 [HuggingFace](https://huggingface.co/hywu/Qwen2idae-16x14B-v1.0)


## Performance
| Model | Activated Params | MMLU (5shot) | GSM8k (5shot) | MATH (4shot) | HumanEval (0shot) | MBPP (4shot) | HellaSwag (10shot) |
|:-----:|:----------------:|:------------:|:-------------:|:------------:|:-----------------:|:------------:|:------------------:|
| GPT3.5 | - | 70.0% | 57.1% | <font color=#F67F70>**34.1%**</font> | <font color=#FBD98D>**48.1%**</font> | - | <font color=#7FEA9E>**85.5%**</font> |
| LLaMA2-70B-chat | 70B | 63.8% | 59.3% | 10.4% | 32.3% | 35.6% | 84.8% |
| Camelidae-8x34B-pro | 35B | <font color=#7FEA9E>**75.7%**</font> | <font color=#F67F70>**79.4%**</font> | <font color=#FBD98D>**24.0%**</font> | <font color=#7FEA9E>**48.8%**</font> | <font color=#7FEA9E>**43.2%**</font> | 85.2% |
| Camelidae-8x34B | 35B | <font color=#FBD98D>**75.6%**</font> | <font color=#7FEA9E>**78.3%**</font> | 22.6% | 43.9% | <font color=#FBD98D>**41.4%**</font> | <font color=#FBD98D>**85.3%**</font> |
| SUSChat-34B | 34B | <font color=#F67F70>**76.4%**</font> | 72.3% | 22.0% | 11.6% | 40.2% | 83.9% |
| Yi-34B-chat | 34B | 74.8% | 67.6% | 17.3% | 20.1% | 41.0% | 83.9% |
| Qwen2idae-16x14B-v1.0 | 15B | 66.7% | <font color=#FBD98D>**77.8%**</font> | <font color=#7FEA9E>**29.9%**</font> | <font color=#F67F70>**62.8%**</font> | <font color=#F67F70>**48.6%**</font> | 82.3% |
| Mixtral-8x7B-instruct | 14B | 68.7% | 71.7% | 22.1% | 25.6% | 40.6% | <font color=#F67F70>**86.5%**</font> |
| Camelidae-8x13B | 13B | 54.4% | 52.6% | 9.8% | 30.6% | 30.4% | 82.5% |
| LLaMA2-13B-chat | 13B | 53.9% | 37.1% | 5.2% | 18.9% | 27.2% | 81.9% |
| Camelidae-8x7B | 7B | 48.3% | 44.0% | 5.8% | 18.3% | 23.4% | 79.2% |
| LLaMA2-7B-chat | 7B | 47.2% | 26.3% | 3.9% | 12.2% | 17.6% | 78.6% |

We bold the top3 scores separately for all models.


## Usage

### Camelidae
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hywu/Camelidae-8x34B", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("hywu/Camelidae-8x34B", device_map="auto", trust_remote_code=True).eval()

inputs = tokenizer('### Human:\nHow are you?\n### Assistant:\n', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```

### Qwen2idae
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hywu/Qwen2idae-16x14B-v1.0", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("hywu/Qwen2idae-16x14B-v1.0", device_map="auto", trust_remote_code=True).eval()

inputs = tokenizer('<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```

## Citation
```bibtex
@article{wu2024parameter,
  title={Parameter-Efficient Sparsity Crafting from Dense to Mixture-of-Experts for Instruction Tuning on General Tasks},
  author={Wu, Haoyuan and Zheng, Haisheng and He, Zhuolun and Yu, Bei},
  journal={arXiv preprint arXiv:2401.02731},
  year={2024}
}
```

## License
The source code in this repo is licensed under the [Apache 2.0 License](https://github.com/wuhy68/Parameter-Efficient-MoE/blob/master/LICENSE). Camelidae and Qwen2idae models are developed for academic research and free commercial use, all usage must adhere to the license from [facebookresearch](https://github.com/facebookresearch/llama/blob/main/LICENSE), [01-ai](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt) and [Qwen1.5](https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/LICENSE).
