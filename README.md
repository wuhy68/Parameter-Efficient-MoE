# Parameter-Efficient Sparsity Crafting From Dense to Mixture-of-Experts for Instruction Tuning on General Tasks

<a href="https://github.com/wuhy68/Parameter-Efficient-MoE/blob/master/LICENSE">
  <img src="https://img.shields.io/badge/Code_License-Apache_2.0-lightblue">
</a>
<a href="https://huggingface.co/hywu">
  <img src="https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg">
</a>

<hr>

## News
- 1/10/2024 - Released the [Camelidae](https://huggingface.co/hywu) models.
- 1/4/2024 - Released the [paper](https://arxiv.org/abs/2401.02731).
- 12/22/2023 - Released the training codes to craft the dense model with LLaMA architecture to the MoE model.

## Introduction
We present Parameter-Efficient Sparsity Crafting to help dense models learn knowledge from different fields (including code and math). This appraoch perfrom instruction tuning and utilize MoE structure in an efficient way.

Parameter-Efficient Sparsity Crafting utilizes parameter efficient techiniques including [QLoRA](https://arxiv.org/abs/2305.14314) and [Adapter](https://arxiv.org/abs/1902.00751) to perfrom Efficient [Sparse Upcycling](https://arxiv.org/abs/2212.05055).

The repo supports the training of dense model using LLaMA architecture ([LLaMA 2](https://arxiv.org/abs/2307.09288), [Yi](https://huggingface.co/01-ai), etc.)

## Model Lists
| Model | Download  
|---|---
Camelidae-8x7B   | [🤗 Hugging Face](https://huggingface.co/hywu/Camelidae-8x7B)
Camelidae-8x13B  | [🤗 Hugging Face](https://huggingface.co/hywu/Camelidae-8x13B)
Camelidae-8x34B  | [🤗 Hugging Face](https://huggingface.co/hywu/Camelidae-8x34B) 

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
[Apache 2.0 License](https://github.com/wuhy68/Parameter-Efficient-MoE/blob/master/LICENSE).