import torch
import torch.nn as nn
from layers.Pyraformer_EncDec import Encoder


class Model(nn.Module):
    """
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """

    def __init__(self, configs, window_size=[4, 4], inner_size=5):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model


        self.encoder = Encoder(configs, window_size, inner_size)


        self.act = torch.nn.functional.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
                (len(window_size) + 1) * self.d_model * configs.seq_len, configs.num_class)




    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # enc
        enc_out = self.encoder(x_enc, x_mark_enc=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)

        return output