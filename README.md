# Parameter-Efficient Sparsity Crafting From Dense to Mixture-of-Experts for Instruction Tuning on General Tasks

<a href="https://github.com/wuhy68/Parameter-Efficient-MoE/blob/master/LICENSE">
  <img src="https://img.shields.io/badge/Code_License-Apache_2.0-lightblue">
</a>
<a href="https://huggingface.co/hywu">
  <img src="https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg">
</a>

## News
- 1/10/2024 - Camelidae models are now available on [🤗HuggingFace](https://huggingface.co/hywu).
- 1/4/2024 - We released the paper, [Parameter-Efficient Sparsity Crafting From Dense to Mixture-of-Experts for Instruction Tuning on General Tasks](https://arxiv.org/abs/2401.02731).
- 12/22/2023 - We released the training repo that craft the dense model with LLaMA architecture to the MoE model.

## Introduction
We present Parameter-Efficient Sparsity Crafting to help dense models learn knowledge from different fields (including code and math). This appraoch perfrom instruction tuning and utilize MoE structure in an efficient way.

Parameter-Efficient Sparsity Crafting utilizes parameter efficient techiniques including [QLoRA](https://arxiv.org/abs/2305.14314) and [Adapter](https://arxiv.org/abs/1902.00751) to perfrom Efficient [Sparse Upcycling](https://arxiv.org/abs/2212.05055).

The repo supports the training of dense model using LLaMA architecture ([LLaMA 2](https://arxiv.org/abs/2307.09288), [Yi](https://huggingface.co/01-ai), etc.)

## Model Lists
| Model | Download  
|---|---
Camelidae-8x7B   | [🤗HuggingFace](https://huggingface.co/hywu/Camelidae-8x7B)
Camelidae-8x13B  | [🤗HuggingFace](https://huggingface.co/hywu/Camelidae-8x13B)
Camelidae-8x34B  | [🤗HuggingFace](https://huggingface.co/hywu/Camelidae-8x34B) 

## Performance
| Model | MMLU (5shot) | GSM8k (5shot) | MATH (4shot) | HumanEval (0shot) | MBPP (4shot) | HellaSwag (10shot) | TriviaQA (0shot) |
|----------------------:|:------------:|:-------------:|:------------:|:-----------------:|:------------:|:------------------:|:----------------:|
| GPT3.5 | 70.0% | 57.1% | **34.1%** | **48.1%** | - | 85.5% | - |
| Camelidae-8x34B | 75.6% | **78.3%** | **22.6%** | **43.9%** | **41.4%** | 85.3% | **63.4%** |
| SUSChat-34B | **76.4%** | 72.3% | 22.0% | 11.6% | 40.2% | 83.9% | 56.1% |
| Mixtral-8x7B-instruct | 68.7% | 71.7% | 22.1% | 25.6% | 40.6% | **86.5%** | 57.7% |
| LLaMA2-70B-chat | 63.8% | 59.3% | 10.4% | 32.3% | 35.6% | 84.8% | 63.0% |
| Camelidae-8x13B | 54.4% | 52.6% | 9.8% | 30.6% | 30.4% | 82.5% | 59.4% |
| LLaMA2-13B-chat | 53.9% | 37.1% | 5.2% | 18.9% | 27.2% | 81.9% | 55.0% |
| Camelidae-8x7B | 48.3% | 44.0% | 5.8% | 18.3% | 23.4% | 79.2% | 51.0% |
| LLaMA2-7B-chat | 47.2% | 26.3% | 3.9% | 12.2% | 17.6% | 78.6% | 46.4% |

We bold the highest scores for open-source models and all models separately.


## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("hywu/Camelidae-8x7B", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("hywu/Camelidae-8x13B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("hywu/Camelidae-8x34B", trust_remote_code=True)

# model = AutoModelForCausalLM.from_pretrained("hywu/Camelidae-8x7B", device_map="auto", trust_remote_code=True).eval()
# model = AutoModelForCausalLM.from_pretrained("hywu/Camelidae-8x13B", device_map="auto", trust_remote_code=True).eval()
model = AutoModelForCausalLM.from_pretrained("hywu/Camelidae-8x34B", device_map="auto", trust_remote_code=True).eval()

inputs = tokenizer('### Human:\nHow are you?\n ### Assistant:\n', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```

## Citation
```bibtex
@article{wu2024parameter,
  title={Parameter-Efficient Sparsity Crafting from Dense to Mixture-of-Experts for Instruction Tuning on General Tasks},
  author={Wu, Haoyuan and Zheng, Haisheng and Yu, Bei},
  journal={arXiv preprint arXiv:2401.02731},
  year={2024}
}
```

## License
The source code in this repo is licensed under the [Apache 2.0 License](https://github.com/wuhy68/Parameter-Efficient-MoE/blob/master/LICENSE). Camelidae models are developed for academic research and free commercial use, all usage must adhere to the license from [facebookresearch](https://github.com/facebookresearch/llama/blob/main/LICENSE) and [01-ai](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt).