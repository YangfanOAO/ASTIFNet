import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import normalize_batch_ts
import warnings
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from natsort import natsorted

warnings.filterwarnings("ignore")


class APAVALoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)

        all_ids = list(data_list[:, 1])  # id of all samples

        val_ids = [1, 2, 17, 18]  # 1, 17 are AD; 2, 18 are HC
        test_ids = [15, 16, 19, 20]  # 15, 19 are AD; 16, 20 are HC

        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]

        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y = self.load_apava(self.data_path, self.label_path, flag=flag)
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_apava(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = natsorted(os.listdir(data_path))
        subject_label = np.load(label_path)

        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", len(ids))
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", len(ids))
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", len(ids))
        else:
            ids = subject_label[:, 1]
            print("all ids:", len(ids))

        for filename, trial_label in zip(filenames, subject_label):
            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)
            if trial_label[1] in ids:
                feature_list.extend(subject_feature)
                label_list.extend([trial_label[0]] * len(subject_feature))

        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class TDBRAINLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)
        all_ids = list(data_list[:, 1])

        val_ids = [18, 19, 20, 21] + [46, 47, 48, 49]  # 8 patients, 4 positive 4 healthy
        test_ids = [22, 23, 24, 25] + [50, 51, 52, 53]  # 8 patients, 4 positive 4 healthy

        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]

        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y = self.load_tdbrain(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_tdbrain(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = natsorted(os.listdir(data_path))
        subject_label = np.load(label_path)

        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", len(ids))
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", len(ids))
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", len(ids))
        else:
            ids = subject_label[:, 1]
            print("all ids:", len(ids))

        for filename, trial_label in zip(filenames, subject_label):
            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)
            if trial_label[1] in ids:
                feature_list.extend(subject_feature)
                label_list.extend([trial_label[0]] * len(subject_feature))

        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)
        # print(X.shape)

        return X, y

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADFTDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y = self.load_adfd(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = bandpass_filter_func(self.X, fs=256, lowcut=0.5, highcut=45)
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):

        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ftd_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Frontotemporal Dementia IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs

        train_ids = (
                cn_list[: int(a * len(cn_list))]
                + ftd_list[: int(a * len(ftd_list))]
                + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
                cn_list[int(a * len(cn_list)): int(b * len(cn_list))]
                + ftd_list[int(a * len(ftd_list)): int(b * len(ftd_list))]
                + ad_list[int(a * len(ad_list)): int(b * len(ad_list))]
        )
        test_ids = (
                cn_list[int(b * len(cn_list)):]
                + ftd_list[int(b * len(ftd_list)):]
                + ad_list[int(b * len(ad_list)):]
        )

        return train_ids, val_ids, test_ids

    def load_adfd(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = natsorted(os.listdir(data_path))
        subject_label = np.load(label_path)

        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", len(ids))
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", len(ids))
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", len(ids))
        else:
            ids = subject_label[:, 1]
            print("all ids:", len(ids))

        for filename, trial_label in zip(filenames, subject_label):
            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)
            if trial_label[1] in ids:
                feature_list.extend(subject_feature)
                label_list.extend([trial_label[0]] * len(subject_feature))

        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class PTBLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptb(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):

        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        my_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial infarction IDs

        train_ids = hc_list[: int(a * len(hc_list))] + my_list[: int(a * len(my_list))]
        val_ids = (
                hc_list[int(a * len(hc_list)): int(b * len(hc_list))]
                + my_list[int(a * len(my_list)): int(b * len(my_list))]
        )
        test_ids = hc_list[int(b * len(hc_list)):] + my_list[int(b * len(my_list)):]

        return train_ids, val_ids, test_ids

    def load_ptb(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = natsorted(os.listdir(data_path))
        subject_label = np.load(label_path)

        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", len(ids))
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", len(ids))
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", len(ids))
        else:
            ids = subject_label[:, 1]
            print("all ids:", len(ids))

        for filename, trial_label in zip(filenames, subject_label):
            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)
            if trial_label[1] in ids:
                feature_list.extend(subject_feature)
                label_list.extend([trial_label[0]] * len(subject_feature))

        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class PTBXLLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptbxl(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):

        data_list = np.load(label_path)
        no_list = list(
            data_list[np.where(data_list[:, 0] == 0)][:, 1]
        )  # Normal ECG IDs
        mi_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial Infarction IDs
        sttc_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # ST/T Change IDs
        cd_list = list(
            data_list[np.where(data_list[:, 0] == 3)][:, 1]
        )  # Conduction Disturbance IDs
        hyp_list = list(
            data_list[np.where(data_list[:, 0] == 4)][:, 1]
        )  # Hypertrophy IDs

        train_ids = (
                no_list[: int(a * len(no_list))]
                + mi_list[: int(a * len(mi_list))]
                + sttc_list[: int(a * len(sttc_list))]
                + cd_list[: int(a * len(cd_list))]
                + hyp_list[: int(a * len(hyp_list))]
        )
        val_ids = (
                no_list[int(a * len(no_list)): int(b * len(no_list))]
                + mi_list[int(a * len(mi_list)): int(b * len(mi_list))]
                + sttc_list[int(a * len(sttc_list)): int(b * len(sttc_list))]
                + cd_list[int(a * len(cd_list)): int(b * len(cd_list))]
                + hyp_list[int(a * len(hyp_list)): int(b * len(hyp_list))]
        )
        test_ids = (
                no_list[int(b * len(no_list)):]
                + mi_list[int(b * len(mi_list)):]
                + sttc_list[int(b * len(sttc_list)):]
                + cd_list[int(b * len(cd_list)):]
                + hyp_list[int(b * len(hyp_list)):]
        )

        return train_ids, val_ids, test_ids

    def load_ptbxl(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = natsorted(os.listdir(data_path))
        subject_label = np.load(label_path)

        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", len(ids))
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", len(ids))
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", len(ids))
        else:
            ids = subject_label[:, 1]
            print("all ids:", len(ids))

        for filename, trial_label in zip(filenames, subject_label):
            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)
            if trial_label[1] in ids:
                feature_list.extend(subject_feature)
                label_list.extend([trial_label[0]] * len(subject_feature))

        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class FLAAPLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/feature.npy')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        self.X, self.y = self.load_flaap_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_flaap_dependent(self, data_path, label_path, flag=None):

        X_train = np.load(data_path)
        y_train = np.load(label_path)
        # print(X_train.shape, y_train.shape)

        # 60 : 20 : 20
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train
        elif flag == 'VAL':
            return X_val, y_val
        elif flag == 'TEST':
            return X_test, y_test
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
            torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class UCIHARLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/feature.npy')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        self.X, self.y = self.load_har_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_har_dependent(self, data_path, label_path, flag=None):

        X_train = np.load(data_path)
        y_train = np.load(label_path)
        # print(X_train.shape, y_train.shape)

        X_test = X_train[-2947:]
        y_test = y_train[-2947:]

        X_train, X_val, y_train, y_val = train_test_split(X_train[:-2947], y_train[:-2947], test_size=0.2,
                                                          random_state=42)

        # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train
        elif flag == 'VAL':
            return X_val, y_val
        elif flag == 'TEST':
            return X_test, y_test
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
            torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class SYNTHETICLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/feature.npy')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        self.X, self.y = self.load_syn_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_syn_dependent(self, data_path, label_path, flag=None):

        X_train = np.load(data_path)
        y_train = np.load(label_path)
        # print(X_train.shape, y_train.shape)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train
        elif flag == 'VAL':
            return X_val, y_val
        elif flag == 'TEST':
            return X_test, y_test
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
            torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


if __name__ == '__main__':
    args = None

    datasets = ['APAVA', 'TDBRAIN', 'ADFTD', 'PTB', 'PTBXL', 'FLAAP', 'UCIHAR', 'SYNTHETIC']
    path = ['APAVA', 'TDBRAIN', 'ADFTD', 'PTB', 'PTB-XL', 'FLAAP', 'UCI-HAR', 'SYNTHETIC']


    for i, dataset in enumerate(datasets):
        root_path = '../dataset/' + path[i]

        train = globals()[f"{dataset}Loader"](args, root_path, flag='TRAIN')
        labels, counts = np.unique(train.y, return_counts=True)
        print(counts)
        val = globals()[f"{dataset}Loader"](args, root_path, flag='VAL')
        labels, counts = np.unique(val.y, return_counts=True)
        print(counts)
        test = globals()[f"{dataset}Loader"](args, root_path, flag='TEST')
        labels, counts = np.unique(test.y, return_counts=True)
        print(counts)

        print(
            f"{dataset} Shape:{train.X.shape[1], train.X.shape[2]} Train: {len(train)}, Valid: {len(val)}, Test: {len(test)}")


