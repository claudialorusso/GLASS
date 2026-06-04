from impl import models, SubGDataset, train, metrics, utils, config
import datasets
import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import argparse
import torch.nn as nn
import functools
import numpy as np
import time
import random
import yaml
from pathlib import Path

test_start = time.time()
parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='ppi_bp')
# Node feature settings.
# deg means use node degree. one means use homogeneous embeddings.
# nodeid means use pretrained node embeddings in ./Emb
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
# node label settings
parser.add_argument('--use_maxzeroone', action='store_true')

parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')
# new args to use checkpoints
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint_every', type=int, default=10)
parser.add_argument('--report_every', type=int, default=10)

args = parser.parse_args()
config.set_device(args.device)


def set_seed(seed: int):
    print("seed ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu

def get_feature_tag():
    if args.use_deg:
        return "use_deg"
    if args.use_one:
        return "use_one"
    if args.use_nodeid:
        return "use_nodeid"
    return "unknown_feature"


def get_checkpoint_paths(repeat, hidden_dim, conv_layer, batch_size):
    feature_tag = get_feature_tag()
    ckpt_dir = Path("checkpoints") / args.dataset / feature_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base_name = (
        f"glass_{args.dataset}_{feature_tag}"
        f"_hidden{hidden_dim}_layers{conv_layer}"
        f"_batch{batch_size}_repeat{repeat}"
    )

    latest_path = ckpt_dir / f"{base_name}_latest.pt"
    best_path = ckpt_dir / f"{base_name}_best.pt"

    return ckpt_dir, latest_path, best_path, base_name


def save_training_checkpoint(
        path,
        iteration,
        gnn,
        optimizer,
        scheduler,
        val_score,
        tst_score,
        early_stop,
        trn_time,
        repeat,
        extra=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(".tmp.pt")

    checkpoint = {
        "iteration": iteration,
        "gnn_state_dict": gnn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_score": val_score,
        "tst_score": tst_score,
        "early_stop": early_stop,
        "trn_time": trn_time,
        "repeat": repeat,
        "args": vars(args),
        "extra": extra or {},
    }

    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)

    print(f"[CHECKPOINT] Saved checkpoint at iter {iteration}: {path}", flush=True)


def load_training_checkpoint(path, gnn, optimizer, scheduler):
    path = Path(path)

    print(f"[CHECKPOINT] Loading checkpoint: {path}", flush=True)
    checkpoint = torch.load(path, map_location=config.device)

    gnn.load_state_dict(checkpoint["gnn_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_iter = checkpoint["iteration"] + 1
    val_score = checkpoint.get("val_score", 0)
    tst_score = checkpoint.get("tst_score", 0)
    early_stop = checkpoint.get("early_stop", 0)
    trn_time = checkpoint.get("trn_time", [])

    print(
        f"[CHECKPOINT] Resuming from iter {start_iter}. "
        f"val={val_score:.4f}, tst={tst_score:.4f}, early_stop={early_stop}",
        flush=True
    )

    return start_iter, val_score, tst_score, early_stop, trn_time

if args.use_seed:
    set_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

baseG = datasets.load_dataset(args.dataset)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn = None

if baseG.y.unique().shape[0] == 2:
    # binary classification task
    def loss_fn(x, y):
        return BCEWithLogitsLoss()(x.flatten(), y.flatten())

    baseG.y = baseG.y.to(torch.float)
    if baseG.y.ndim > 1:
        output_channels = baseG.y.shape[1]
    else:
        output_channels = 1
    score_fn = metrics.aml_metrics #metrics.binaryf1
else:
    # multi-class classification task
    baseG.y = baseG.y.to(torch.int64)
    loss_fn = CrossEntropyLoss()
    output_channels = baseG.y.unique().shape[0]
    score_fn = metrics.microf1

loader_fn = SubGDataset.GDataloader
tloader_fn = SubGDataset.GDataloader


def split():
    '''
    load and split dataset.
    '''
    # initialize and split dataset
    global trn_dataset, val_dataset, tst_dataset, baseG
    global max_deg, output_channels, loader_fn, tloader_fn
    baseG = datasets.load_dataset(args.dataset)
    if baseG.y.unique().shape[0] == 2:
        baseG.y = baseG.y.to(torch.float)
    else:
        baseG.y = baseG.y.to(torch.int64)
    # initialize node features
    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        raise NotImplementedError

    max_deg = torch.max(baseG.x)
    baseG.to(config.device)
    # split data
    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))#(*baseG.get_split("test"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))#(*baseG.get_split("valid"))
    print("TRAIN dataset length:", len(trn_dataset))
    print("VAL dataset length:", len(val_dataset))
    print("TEST dataset length:", len(tst_dataset))
    # choice of dataloader
    if args.use_maxzeroone:

        def tfunc(ds, bs, shuffle=True, drop_last=True):
            return SubGDataset.ZGDataloader(ds,
                                            bs,
                                            z_fn=utils.MaxZOZ,
                                            shuffle=shuffle,
                                            drop_last=drop_last)

        def loader_fn(ds, bs):
            return tfunc(ds, bs, shuffle=True, drop_last=False)

        def tloader_fn(ds, bs):
            return tfunc(ds, bs, True, False)
    else:

        def loader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs)

        def tloader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs, shuffle=True)


def buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio, aggr):
    '''
    Build a GLASS model.
    Args:
        jk: whether to use Jumping Knowledge Network.
        conv_layer: number of GLASSConv.
        pool: pooling function transfer node embeddings to subgraph embeddings.
        z_ratio: see GLASSConv in impl/model.py. Z_ratio in [0.5, 1].
        aggr: aggregation method. mean, sum, or gcn.
    '''
    conv = models.EmbZGConv(hidden_dim,
                            hidden_dim,
                            conv_layer,
                            max_deg=max_deg,
                            activation=nn.ELU(inplace=True),
                            jk=jk,
                            dropout=dropout,
                            conv=functools.partial(models.GLASSConv,
                                                   aggr=aggr,
                                                   z_ratio=z_ratio,
                                                   dropout=dropout),
                            gn=True)

    # use pretrained node embeddings.
    if args.use_nodeid:
        print("load ", f"./Emb/{args.dataset}_{hidden_dim}.pt")
        emb = torch.load(f"./Emb/{args.dataset}_{hidden_dim}.pt",
                         map_location=torch.device('cpu')).detach()
        conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)

    mlp = nn.Linear(hidden_dim * (conv_layer) if jk else hidden_dim,
                    output_channels)

    pool_fn_fn = {
        "mean": models.MeanPool,
        "max": models.MaxPool,
        "sum": models.AddPool,
        "size": models.SizePool
    }
    if pool in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool]()
    else:
        raise NotImplementedError

    gnn = models.GLASS(conv, torch.nn.ModuleList([mlp]),
                       torch.nn.ModuleList([pool_fn1])).to(config.device)
    return gnn


def test(pool="size",
         aggr="mean",
         hidden_dim=64,
         conv_layer=8,
         dropout=0.3,
         jk=1,
         lr=1e-3,
         z_ratio=0.8,
         batch_size=None,
         resi=0.7):
    '''
    Test a set of hyperparameters in a task.
    Args:
        jk: whether to use Jumping Knowledge Network.
        z_ratio: see GLASSConv in impl/model.py. A hyperparameter of GLASS.
        resi: the lr reduce factor of ReduceLROnPlateau.
    '''
    outs = []
    t1 = time.time()
    # we set batch_size = tst_dataset.y.shape[0] // num_div.
    num_div = tst_dataset.y.shape[0] / batch_size
    # we use num_div to calculate the number of iteration per epoch and count the number of iteration.
    if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
        num_div /= 5

    outs = []
    for repeat in range(args.repeat):
        set_seed((1 << repeat) - 1)
        print(f"repeat {repeat}")
        split()
        gnn = buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio,
                         aggr)
        trn_loader = loader_fn(trn_dataset, batch_size)
        val_loader = tloader_fn(val_dataset, batch_size)
        tst_loader = tloader_fn(tst_dataset, batch_size)
        print("TRAIN loader length:", len(trn_loader))
        print("VAL loader length:", len(val_loader))
        print("TEST loader length:", len(tst_loader))
        optimizer = Adam(gnn.parameters(), lr=lr)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=resi,
                                             min_lr=5e-5)
        ckpt_dir, latest_ckpt, best_ckpt, base_ckpt_name = get_checkpoint_paths(
        repeat=repeat,
        hidden_dim=hidden_dim,
        conv_layer=conv_layer,
        batch_size=batch_size,
        )
        val_score = 0
        tst_score = 0
        early_stop = 0
        trn_time = []
        start_iter = 0
        if args.resume and latest_ckpt.exists():
          start_iter, val_score, tst_score, early_stop, trn_time = load_training_checkpoint(
            latest_ckpt,
            gnn,
            optimizer,
            scd,
          )
        elif args.resume:
          print(f"[CHECKPOINT] Resume requested but checkpoint not found: {latest_ckpt}", flush=True)
        for i in range(start_iter, 300):
            t1 = time.time()
            loss = train.train(optimizer, gnn, trn_loader, loss_fn)
            trn_time.append(time.time() - t1)
            scd.step(loss)

            if i >= 0:#if i >= 100 / num_div:
                score, _ = train.test(gnn,
                                      val_loader,
                                      score_fn,
                                      loss_fn=loss_fn)

                if score > val_score:
                    early_stop = 0
                    val_score = score
                    score, _ = train.test(gnn,
                                          tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn)
                    tst_score = score
                    print(
                        f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}",
                        flush=True)
                    save_training_checkpoint(
                        best_ckpt,
                        i,
                        gnn,
                        optimizer,
                        scd,
                        val_score,
                        tst_score,
                        early_stop,
                        trn_time,
                        repeat,
                        extra={
                            "checkpoint_type": "best_val",
                            "loss": float(loss),
                        }
                    )
                elif score >= val_score - 1e-5:
                    score, _ = train.test(gnn,
                                          tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn)
                    tst_score = max(score, tst_score)
                    print(
                        f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {score:.4f}",
                        flush=True)
                else:
                    early_stop += 1
                    if i % 10 == 0:
                        print(
                            f"iter {i} loss {loss:.4f} val {score:.4f} tst {train.test(gnn, tst_loader, score_fn, loss_fn=loss_fn)[0]:.4f}",
                            flush=True)
            if args.checkpoint_every > 0 and (i + 1) % args.checkpoint_every == 0:
                save_training_checkpoint(
                    latest_ckpt,
                    i,
                    gnn,
                    optimizer,
                    scd,
                    val_score,
                    tst_score,
                    early_stop,
                    trn_time,
                    repeat,
                    extra={
                        "loss": float(loss),
                        "hidden_dim": hidden_dim,
                        "conv_layer": conv_layer,
                        "batch_size": batch_size,
                        "pool": pool,
                        "aggr": aggr,
                        "dropout": dropout,
                        "jk": jk,
                        "lr": lr,
                        "z_ratio": z_ratio,
                        "resi": resi,
                    }
                )

            if args.report_every > 0 and (i + 1) % args.report_every == 0:
                print(f"\nPERIODIC TEST REPORT - iter {i}", flush=True)
                train.test_full_report(
                    gnn,
                    tst_loader,
                    metrics.aml_metrics_report,
                    loss_fn=loss_fn,
                )

            if val_score >= 1 - 1e-5:
                early_stop += 1
            if early_stop > 100 / num_div:
                break
        """
        print(
            f"end: epoch {i+1}, train time {sum(trn_time):.2f} s, val {val_score:.3f}, tst {tst_score:.3f}",
            flush=True)
        """
        print("\nFINAL TEST REPORT")
        final_report = train.test_full_report(
            gnn,
            tst_loader,
            metrics.aml_metrics_report,
            loss_fn=loss_fn,
        )

        print(
            f"end: epoch {i + 1}, train time {sum(trn_time):.2f} s, val {val_score:.3f}, tst {tst_score:.3f}",
            flush=True)


        final_ckpt = ckpt_dir / f"{base_ckpt_name}_final.pt"
        save_training_checkpoint(
            final_ckpt,
            i,
            gnn,
            optimizer,
            scd,
            val_score,
            tst_score,
            early_stop,
            trn_time,
            repeat,
            extra={
                "checkpoint_type": "final",
            }
        )
        outs.append(tst_score)
    print(
        f"average {np.average(outs):.3f} error {np.std(outs) / np.sqrt(len(outs)):.3f}"
    )


print(args)
# read configuration
with open(f"config/{args.dataset}.yml") as f:
    params = yaml.safe_load(f)

print("params", params, flush=True)
split()
test(**(params))
tot_time = time.time() - test_start
print(f"\033[33mTotal runtime: {tot_time:.2f}s\033[0m")
print(f"\033[33mTotal runtime: {tot_time/ 60:.2f}min\033[0m")
