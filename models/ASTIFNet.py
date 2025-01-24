import torch
import torch.nn as nn
import torch.nn.init as init
from thop import profile
from thop import clever_format


class ATT(torch.nn.Module):
    def __init__(self, e_lambda=1e-4,bias=0.5):
        super(ATT, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.bias = bias

    def forward(self, x):
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        v = torch.var(x, dim=[2, 3], keepdim=True) + self.e_lambda

        y = d / v + self.bias

        return x * self.act(y)

class ASTIF(nn.Module):

    def __init__(self, channel_size, emb_size, kernel_size_t, kernel_size_s):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(channel_size, emb_size, kernel_size=(1, kernel_size_t)),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, kernel_size=(kernel_size_s, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ReLU()
        )
        self.att = ATT()

        self.initialize_weights()

    def forward(self, x):

        x = self.temporal(x)

        x = self.spatial(x)

        x = self.att(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_class = configs.num_class

        self.hidden = configs.emb_size
        self.input = [1] + self.hidden[:-1]
        self.kernel_t = configs.kernel_t
        self.kernel_s = configs.kernel_s

        self.net = nn.ModuleList([
            ASTIF(i, h, t, s)
            for i, h, t, s in zip(self.input, self.hidden, self.kernel_t, self.kernel_s)
        ])

        self.aap = nn.AdaptiveAvgPool2d((1, 1))

        self.predict = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden[-1], self.num_class),
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        x = x.permute(0, 2, 1).unsqueeze(1)
        for net in self.net:
            x = net(x)

        x = self.aap(x)

        y = self.predict(x)

        return y


class Configs:
    def __init__(self, seq_len, num_class, enc_in, emb_size, kernel_t, kernel_s, dropout):
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.num_class = num_class

        self.emb_size = emb_size
        self.kernel_t = kernel_t
        self.kernel_s = kernel_s

        self.dropout = dropout


if __name__ == '__main__':
    configs = Configs(seq_len=256, num_class=2, enc_in=33, emb_size=[7, 8, 9], kernel_t=[5, 6, 5], kernel_s=[4, 3, 4],
                      dropout=0.1)
    model = Model(configs)
    x = torch.randn((1, configs.seq_len, configs.enc_in))

    y = model(x, None, None, None)
    print(y.shape)

    flops, params = profile(model, inputs=(x, None, None, None))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
