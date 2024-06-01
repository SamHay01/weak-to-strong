from dataclasses import dataclass
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, GemmaForCausalLM
from peft import get_peft_model, PeftModel, PeftConfig

@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):
    """
    This class initializes the linear head to zeros
    """

    def __init__(self, name, linear_probe=False, lora_config=None, **kwargs):
        config = AutoConfig.from_pretrained(name, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels
        lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if lora_config is not None:
            lm = get_peft_model(lm, lora_config)
            
        self.lm = lm
        if name == "google/Gemma-2b":
            self.transformer = lm.model.model.embed_tokens
            head_dtype = lm.lm_head.weight.dtype
        elif name=="google-bert/bert-base-cased":
            self.transformer = lm.bert
            head_dtype = lm.cls.predictions.weight.dtype
        else:
            self.transformer = lm.transformer
            head_dtype = lm.lm_head.weight.dtype
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))
        self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
            head_dtype
        )
        torch.nn.init.normal_(self.score.weight, std=0.0)
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, name, lora_config=None, **kwargs):
        return cls(name, lora_config=lora_config,**kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def forward(self, input_ids: torch.LongTensor):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        input_lens = (input_ids != 0).sum(dim=-1)
        transformer_outputs = self.transformer(input_ids)
        print(f'output shape = {transformer_outputs.shape}')
        hidden_states = torch.stack(
            [transformer_outputs[0][i, input_lens[i] - 1, :] for i in range(len(input_lens))]
        )
        self.score.to(hidden_states.device)
        if self.linear_probe:
            hidden_states = hidden_states.detach()
        logits = self.score(hidden_states)
        return logits
