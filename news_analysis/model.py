import torch
import torch.nn as nn
from transformers import AutoModel

class FusionModel(nn.Module):
    def __init__(self, price_window: int = 5, hidden: int = 16, dropout_prob: float = 0.3):
        super().__init__()
        self.text_enc = AutoModel.from_pretrained("ai-forever/ruBERT-base")
        self.price_enc = nn.LSTM(input_size=1,
                                 hidden_size=hidden,
                                 batch_first=True)
        combined_size = self.text_enc.config.hidden_size + hidden
        self.head = nn.Sequential(
    nn.Linear(combined_size, combined_size//2),
    nn.ReLU(),
    nn.Dropout(dropout_prob),
    nn.Linear(combined_size//2, 2)
)

    def forward(self, ids, prices):
        bert_output = self.text_enc(ids)
        txt_feat = bert_output.last_hidden_state[:, 0, :]
        mean = prices.mean(dim=1, keepdim=True)
        std = prices.std(dim=1, unbiased=False, keepdim=True)
        normed = (prices - mean) / (std + 1e-6)
        _, (h_n, _) = self.price_enc(normed.unsqueeze(-1))
        price_feat = h_n.squeeze(0)
        combined = torch.cat([txt_feat, price_feat], dim=1)
        logits = self.head(combined)
        return logits