import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import argparse
import pandas as pd

plt.switch_backend("agg")


def result_save(metrics_list, run_info, model_name, dataset_name):
    average_metrics = {}
    std_metrics = {}
    metrics_dict = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0, "AUROC": 0, "AUPRC": 0}

    for metric in metrics_dict:
        values = [metric_dict[metric] for metric_dict in metrics_list]
        avg = np.mean(values)
        std = np.std(values)

        average_metrics[metric] = f"{avg * 100:.2f}"
        std_metrics[metric] = f"{std * 100:.2f}"

    folder_path = "./results/"
    file_name = "result.txt"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    width = 7
    setting = model_name + '_' + dataset_name
    with open(os.path.join(folder_path, file_name), "a") as f:
        f.write(setting + "  \n")

        f.write(f"{'Average : ':<{width}}")
        for metric, avg in average_metrics.items():
            f.write(f"{metric}: {avg:<{width}}")
        f.write("\n")

        f.write(f"{'Standard: ':<{width}}")
        for metric, std in std_metrics.items():
            f.write(f"{metric}: {std:<{width}}")
        f.write("\n")
        f.write("\n")

    new_row = [model_name, dataset_name, average_metrics['Accuracy'], average_metrics['Precision'],
               average_metrics['Recall'], average_metrics['F1'], average_metrics['AUROC'], average_metrics['AUPRC']]
    file_path = folder_path + '/result.csv'
    try:
        df = pd.read_csv(file_path)
        df.loc[len(df)] = new_row
        df.to_csv(file_path, index=False)
    except FileNotFoundError:
        columns = ['Classifier', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC', 'AUPRC']
        df = pd.DataFrame([new_row], columns=columns)
        df.to_csv(file_path, index=False)

    new_row = [model_name, dataset_name, average_metrics['Accuracy'], run_info[0], run_info[1], run_info[2]]
    file_path = folder_path + '/run_info.csv'
    try:
        df = pd.read_csv(file_path)
        df.loc[len(df)] = new_row
        df.to_csv(file_path, index=False)
    except FileNotFoundError:
        columns = ['Classifier', 'Dataset', 'Accuracy', 'Flops', 'Params', 'Time']
        df = pd.DataFrame([new_row], columns=columns)
        df.to_csv(file_path, index=False)

    print("mean：", average_metrics)
    print("std：", std_metrics)
    print(run_info)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * (0.85 ** ((epoch - 5) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == "const":
        lr_adjust = {epoch: args.learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Metric score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n"
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
