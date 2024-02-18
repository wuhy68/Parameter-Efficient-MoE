import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from camelidae.configuration_camelidae import CamelidaeConfig
from camelidae.modeling_camelidae import LlamaForCausalLM

from peft import PeftModel

import torch


def merge_lora_to_base_model():
    from transformers_utils import get_keys_to_not_convert, _load_pretrained_model
    import transformers.utils.bitsandbytes
    import transformers.modeling_utils

    transformers.utils.bitsandbytes.get_keys_to_not_convert = get_keys_to_not_convert
    transformers.modeling_utils.PreTrainedModel._load_pretrained_model = (
        _load_pretrained_model
    )

    # Adjust to your corresponding path
    model_path = "./"
    peft_path="./adapter_model/"
    moe_path="./moe_model.bin"
    save_path = "./"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    model_config = CamelidaeConfig.from_pretrained(model_path)
    model_config.pretraining_tp = 1  ## without tensor parallelism rank

    # Place the corresponding two files in the save_path
    model_config.auto_map = {
        "AutoConfig": "configuration_camelidae.CamelidaeConfig",
        "AutoModelForCausalLM": "modeling_camelidae.LlamaForCausalLM"
    }

    # Camelidae Config
    model_config.moe_dtype = "bfloat16"
    model_config.adapter_dim = 512
    model_config.topk = 2
    model_config.moe_scaling = 0.25
    model_config.num_experts = 8
    model_config.output_router_logits = False

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map={'': 'cpu'}
    )

    moe_weights = torch.load(moe_path, map_location=torch.device("cpu"))
    weights_dict = {}
    for k, v in moe_weights.items():
        new_k = k.replace("base_model.model.", "") if "base_model.model." in k else k
        weights_dict[new_k] = v

    model.load_state_dict(weights_dict, strict=False)

    model = PeftModel.from_pretrained(
        model,
        peft_path,
        torch_dtype=torch.bfloat16,
        device_map={'': 'cpu'}
    )

    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

def test_loading():

    # Merge model saved path
    path = ""
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params/(1000000000):.2f}B total parameters.')

    inputs = tokenizer('### Human:\nHow are you?\n### Assistant:\n', return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

if __name__ == '__main__':
    merge_lora_to_base_model()
    test_loading()