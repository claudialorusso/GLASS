from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    log_loss,
)
import numpy as np


def binaryf1(pred, label):
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0], -1)
    return f1_score(label_i, pred_i, average="micro")


def microf1(pred, label):
    pred_i = np.argmax(pred, axis=1)
    return f1_score(label, pred_i, average="micro")


def auroc(pred, label):
    return roc_auc_score(label, pred)


def aml_metrics(pred, label):
    """
    Used inside the GLASS training loop as score_fn.
    IMPORTANT: no print here, otherwise validation/test loop becomes noisy.
    Returns binary F1 on positive class = suspicious = 1.
    """
    pred = pred.reshape(-1)
    label = label.reshape(-1).astype(np.int64)

    pred_bin = (pred > 0).astype(np.int64)

    return f1_score(
        label,
        pred_bin,
        average="binary",
        zero_division=0,
    )


def aml_metrics_report(pred, label, loss=None):
    """
    Final AML-oriented report.
    Prints metrics coherently with RE3PY and RevTrack.
    pred = logits
    label = binary labels
    """

    pred = pred.reshape(-1)
    label = label.reshape(-1).astype(np.int64)

    pred_bin = (pred > 0).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        label,
        pred_bin,
        labels=[0, 1],
    ).ravel()

    out = {
        "accuracy": accuracy_score(label, pred_bin),

        "precision_binary": precision_score(label, pred_bin, average="binary", zero_division=0),
        "recall_binary": recall_score(label, pred_bin, average="binary", zero_division=0),
        "f1_binary": f1_score(label, pred_bin, average="binary", zero_division=0),

        "precision_macro": precision_score(label, pred_bin, average="macro", zero_division=0),
        "recall_macro": recall_score(label, pred_bin, average="macro", zero_division=0),
        "f1_macro": f1_score(label, pred_bin, average="macro", zero_division=0),

        "precision_micro": precision_score(label, pred_bin, average="micro", zero_division=0),
        "recall_micro": recall_score(label, pred_bin, average="micro", zero_division=0),
        "f1_micro": f1_score(label, pred_bin, average="micro", zero_division=0),

        "auroc": roc_auc_score(label, pred),
        "prauc": average_precision_score(label, pred),

        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

    if loss is not None:
        out["loss"] = float(loss)

    print("\nC O N F U S I O N  M A T R I X:")
    print(f"True Positives (TP): {out['TP']}")
    print(f"True Negatives (TN): {out['TN']}")
    print(f"False Positives (FP): {out['FP']}")
    print(f"False Negatives (FN): {out['FN']}")

    print("\nA M L  E V A L U A T I O N  M E T R I C S:")
    for k, v in out.items():
        print(f"{k}: {v}")

    return out