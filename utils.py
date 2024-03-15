import datetime
import errno
import os
import pickle
import random
from pprint import pprint

import dgl

import numpy as np
import torch
from dgl.data.utils import _get_dgl_url, download, get_download_dir
import pickle as pkl
from scipy import io as sio, sparse


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print("Directory {} already exists.".format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = "{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args["log_dir"], "{}_{}".format(args["dataset"], date_postfix)
    )

    if sampling:
        log_dir = log_dir + "_sampling"

    mkdir_p(log_dir)
    return log_dir


sampling_configure = {"batch_size": 20}


def setup(args):
    set_random_seed(args["seed"])
    args["device"] = "cuda:1" if torch.cuda.is_available() else "cpu"
    args["log_dir"] = setup_log_dir(args)
    return args


def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    args["log_dir"] = setup_log_dir(args, sampling=True)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_dblp(remove_self_loop):
    assert not remove_self_loop

    data_path = get_download_dir() + "/DBLP.mat"
    if not os.path.isfile (data_path):
        download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)

    label = data['label']
    train_idx = data['train_idx'][0]
    val_idx = data['val_idx'][0]
    test_idx = data['test_idx'][0]
    features = data['features']
    net_APA = data['net_APA']
    net_APTPA = data['net_APTPA']
    net_APCPA = data['net_APCPA']

    hg = dgl.heterograph({
        ('author', 'ap', 'paper'): net_APA.nonzero(),
        ('paper', 'pa', 'author'): net_APA.transpose().nonzero(),
        ('paper', 'pt', 'terms'): net_APTPA.nonzero(),
        ('terms', 'tp', 'paper'): net_APTPA.transpose().nonzero(),
        ('paper', 'pc', 'conference'): net_APCPA.nonzero(),
        ('conference', 'cp', 'paper'): net_APCPA.transpose().nonzero()
    })

    features = torch.FloatTensor(features)
    num_classes = label.shape[1]
    labels = torch.FloatTensor(label)
    labels = torch.argmax(labels, dim=1)

    num_nodes = hg.num_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    meta_paths = [["ap", "pa"], ["ap", "pt", "tp", "pa"], ["ap", "pc", "cp","pa"]]

    return (
        hg,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
        meta_paths,
    )

def load_acm(remove_self_loop):
    assert not remove_self_loop
    url = "dataset/ACM.mat"
    data_path = get_download_dir() + "/ACM.mat"
    if not os.path.isfile (data_path):
        download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)

    p_vs_l = data["PvsL"]  # paper-field?
    p_vs_a = data["PvsA"]  # paper-author
    p_vs_t = data["PvsT"]  # paper-term, bag of words
    p_vs_c = data["PvsC"]  # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph(
        {
            ("paper", "pa", "author"): p_vs_a.nonzero(),
            ("author", "ap", "paper"): p_vs_a.transpose().nonzero(),
            ("paper", "pf", "field"): p_vs_l.nonzero(),
            ("field", "fp", "paper"): p_vs_l.transpose().nonzero(),
        }
    )

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = pc_c == conf_id
        float_mask[pc_c_mask] = np.random.permutation(
            np.linspace(0, 1, pc_c_mask.sum())
        )
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.num_nodes("paper")
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    meta_paths = [["pa", "ap"], ["pf", "fp"]]

    return (
        hg,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
        meta_paths,
    )

def load_imdb (remove_self_loop):
    assert not remove_self_loop

    data_path = get_download_dir() + "/imdb.pkl"
    if not os.path.isfile (data_path):
        download(_get_dgl_url(url), path=data_path)

    data = pkl.load(open(data_path, "rb"))

    label = data['label']
    train_idx = data['train_idx'][0]
    val_idx = data['val_idx'][0]
    test_idx = data['test_idx'][0]
    features = data['feature']
    MAM = data['MAM']
    MDM = data['MDM']

    hg = dgl.heterograph(
        {
            ("movie", "ma", "actor"): MAM.nonzero(),
            ("actor", "am", "movie"): MAM.transpose().nonzero(),
            ("movie", "md", "director"): MDM.nonzero(),
            ("director", "dm", "movie"): MDM.transpose().nonzero(),
        }
    )

    features = torch.FloatTensor(features)
    num_classes = label.shape[1]
    labels = torch.FloatTensor(label)
    labels = torch.argmax(labels, dim=1)

    num_nodes = hg.num_nodes('actor')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    meta_paths = [["ma", "am"], ["md", "dm"]]

    return (
        hg,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
        meta_paths,
    )

def load_data(dataset, remove_self_loop=False):
    if dataset == "ACM":
        return load_acm(remove_self_loop)
    elif dataset == "DBLP":
        return load_dblp(remove_self_loop)
    elif dataset == "IMDB":
        return load_imdb(remove_self_loop)
    else:
        raise NotImplementedError("Unsupported dataset {}".format(dataset))


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = "stop/early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
