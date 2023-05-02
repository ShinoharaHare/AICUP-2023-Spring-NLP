import fire
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import LlamaModel


def main(
    model_path: str,
    checkpoint_path: str,
    output_path: str,
):
    lora_config = LoraConfig(
        r=8,
        target_modules=['q_proj', 'v_proj'],
        lora_alpha=16,
        lora_dropout=0.05
    )

    base_model = LlamaModel.from_pretrained(model_path, device_map='auto', load_in_8bit=True, low_cpu_mem_usage=True)
    model = get_peft_model(base_model, lora_config)
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        state_dict[k.removeprefix('base_model.')] = state_dict[k]
    set_peft_model_state_dict(model, state_dict)
    model.save_pretrained(output_path)


if __name__ == '__main__':
    fire.Fire(main)
