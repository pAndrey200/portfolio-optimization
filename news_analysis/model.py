import torch
import torch.nn as nn
from transformers import AutoModel

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained("ai-forever/ruBERT-base")
        self.classifier = nn.Linear(self.text_model.config.hidden_size, 2)

    def forward(self, input_ids):
        out = self.text_model(input_ids=input_ids)
        pooled = out.last_hidden_state[:, 0]  # CLS token
        return self.classifier(pooled)