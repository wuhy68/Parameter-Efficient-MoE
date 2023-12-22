# Parameter-Efficient Sparsity Crafting From Dense to Mixture-of-Experts

We utilize parameter efficient techiniques including [QLoRA](https://arxiv.org/abs/2305.14314) and [Adapter](https://arxiv.org/abs/1902.00751) to perfrom Efficient [Sparse Upcycling](https://arxiv.org/abs/2212.05055).

## Updates
- 12/22/2023 - Releases the training codes to craft dense model LLaMA to MoE model.

## Overview
We present Parameter-Efficient Sparsity Crafting to help dense models learn knowledge from different fields (including code and math). This appraoch perfrom instruction tuning and utilize MoE structure in an efficient way.

The repo supports the training of dense model using LLaMA architecture ([LLaMA 2](https://arxiv.org/abs/2307.09288), [Yi](https://huggingface.co/01-ai), etc.)

## Todo
- [x] Release the training code.
- [ ] Support [Qwen](https://huggingface.co/Qwen) series.
- [ ] Release the evaluation results of LLaMA 2 7B and LLaMA 2 13B.
- [ ] Release the checkpoint and training data.

## Citation
```bibtex
@misc{2023pesc,
    title={Parameter-Efficient Sparsity Crafting From Dense to Mixture-of-Experts},
    author={pesc Team},
    howpublished = {\url{https://github.com/wuhy68/Parameter-Efficient-MoE}},
    year={2023}
}
```