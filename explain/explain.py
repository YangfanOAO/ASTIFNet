import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import os
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_provider.data_loader import TDBRAINLoader, PTBLoader, APAVALoader, SYNTHETICLoader
from models import ASTIFNet

data_dict = {"TDBRAIN": TDBRAINLoader, "PTB": PTBLoader, 'APAVA': APAVALoader, 'SYNTHETIC': SYNTHETICLoader}


class Configs:
    def __init__(self, seq_len, num_class, enc_in, emb_size, kernel_t, kernel_s, dropout):
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.num_class = num_class

        self.emb_size = emb_size
        self.kernel_t = kernel_t
        self.kernel_s = kernel_s

        self.dropout = dropout


def draw_line(x, weights, name):
    min_val = np.min(weights)
    max_val = np.max(weights)
    weights = (weights - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(3, 3))
    y_min = 10000
    y_max = -100

    for i in range(len(x)):
        for j in range(1, len(x[i])):
            weight = weights[i][j]
            color = plt.cm.cool(weight)
            offset = (len(x) - i) * 4.5

            ax.plot([j - 1, j], [x[i][j - 1] + offset, x[i][j] + offset], color=color, linewidth=1)
            y_min = min(y_min, x[i][j - 1] + offset)
            y_max = max(y_max, x[i][j - 1] + offset)

    ax.set_yticks([])
    ax.set_xticks([])

    plt.xlim(0, len(x[0]))
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f'{name}_line.png', bbox_inches='tight', dpi=1024, pad_inches=0)

    plt.show()


def cam_line(x, model, name):
    x_ = torch.tensor(x.reshape(1, x.shape[0], x.shape[1]), dtype=torch.float32)

    model.eval()
    final_conv_layer = model.net[-1]

    def get_features_hook(module, input, output):
        global features_map
        features_map = output.cpu()

    final_conv_layer.register_forward_hook(get_features_hook)
    output = model(x_, None, None, None)
    print(output)

    weights = model.predict[-1].weight.data.cpu().numpy()
    pred_class = output.argmax(dim=1).item()
    weights_for_class = weights[pred_class]
    weights_for_class = torch.tensor(weights_for_class)
    print(weights_for_class.shape, features_map.shape)

    cam = torch.matmul(weights_for_class, features_map.view(features_map.size(1), -1))
    cam = cam.view(features_map.size(2), features_map.size(3))
    cam = np.maximum(cam.detach().numpy(), 0)

    cam = cv2.resize(cam, (x.shape[0], x.shape[1]))
    x = x.transpose(1, 0)

    plt.subplots(figsize=(3, 3))
    plt.imshow(x, cmap='viridis', alpha=1, aspect='auto')
    plt.imshow(cam, cmap='jet', alpha=0.4, aspect='auto')

    plt.savefig(f'{name}_cam.png', bbox_inches='tight', pad_inches=0, dpi=1024)
    plt.show()
    plt.close()

    draw_line(x, cam, name)


def apava(id_list):
    name = 'APAVA'
    path = '/root/ASTIFNet/dataset/APAVA/'

    configs = Configs(seq_len=256, num_class=2, enc_in=16, emb_size=[8, 9, 8], kernel_t=[4, 5, 4], kernel_s=[3, 4, 3],
                      dropout=0.1)

    model = ASTIFNet.Model(configs)
    paths = "/root/ASTIFNet/checkpoints/classification/APAVA-Indep/ASTIFNet/classification_APAVA-Indep_ASTIFNet_APAVA_dm512_swaFalse_bc32_lr0.002_type3_seed41/checkpoint.pth"
    model.load_state_dict(torch.load(paths))

    dataset = data_dict[name](None, path, 'TEST')

    for i in id_list:
        x = dataset.X[i, :, :]
        y = int(dataset.y[i])
        cam_line(x, model, f'apava_{i}_{y}')


def ptb(id_list):
    name = 'PTB'
    path = '/root/ASTIFNet/dataset/PTB/'

    configs = Configs(seq_len=256, num_class=2, enc_in=33, emb_size=[11, 18, 11], kernel_t=[7, 4, 5],
                      kernel_s=[4, 5, 4],
                      dropout=0.1)

    model = ASTIFNet.Model(configs)
    paths = "/root/ASTIFNet/checkpoints/classification/PTB-Indep/ASTIFNet/classification_PTB-Indep_ASTIFNet_PTB_dm512_swaFalse_bc128_lr0.002_type3_seed41/checkpoint.pth"
    model.load_state_dict(torch.load(paths))

    dataset = data_dict[name](None, path, 'TEST')

    for i in id_list:
        x = dataset.X[i, :, :]
        y = int(dataset.y[i])
        cam_line(x, model, f'ptb_{i}_{y}')


def syn(id_list):
    name = 'SYNTHETIC'
    path = '/root/ASTIFNet/dataset/SYNTHETIC/'

    configs = Configs(seq_len=256, num_class=2, enc_in=33, emb_size=[32, 32, 32], kernel_t=[4, 5, 4],
                      kernel_s=[2, 2, 2],
                      dropout=0.1)

    model = ASTIFNet.Model(configs)
    paths = "/root/ASTIFNet/checkpoints/addition/SYNTHETIC/ASTIFNet/addition_SYNTHETIC_ASTIFNet_SYNTHETIC_dm512_swaFalse_bc32_lr0.002_type3_seed41/checkpoint.pth"

    model.load_state_dict(torch.load(paths))

    dataset = data_dict[name](None, path, 'TEST')

    for i in id_list:
        x = dataset.X[i, :, :]
        y = int(dataset.y[i])
        cam_line(x, model, f'syn_{i}_{y}')


if __name__ == '__main__':
    apava([0])
    ptb([1])
    syn([0, 1])
