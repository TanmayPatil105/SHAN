import torch
from sklearn.metrics import f1_score
from model_hetero import HAN
from utils import EarlyStopping, load_data


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    # Heterogenous graph
    (
        g,
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
    ) = load_data(args["dataset"])

    if hasattr(torch, "BoolTensor"):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args["device"])
    labels = labels.to(args["device"])
    train_mask = train_mask.to(args["device"])
    val_mask = val_mask.to(args["device"])
    test_mask = test_mask.to(args["device"])

    num_heads = []
    for i in range (0, args["num_heads"]):
        num_heads.append (8)

    model = HAN(
        meta_paths=meta_paths,
        in_size=features.shape[1],
        hidden_size=args["hidden_units"],
        out_size=num_classes,
        num_heads=num_heads,
        dropout=args["dropout"],
    ).to(args["device"])

    g = g.to(args["device"])
    print (g)

    stopper = EarlyStopping(patience=args["patience"])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )

    for epoch in range(args["num_epochs"]):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(
            logits[train_mask], labels[train_mask]
        )
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model, g, features, labels, val_mask, loss_fcn
        )
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print(
            "Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | "
            "Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
                epoch + 1,
                loss.item(),
                train_micro_f1,
                train_macro_f1,
                val_loss.item(),
                val_micro_f1,
                val_macro_f1,
            )
        )

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model, g, features, labels, test_mask, loss_fcn
    )
    print(
        "Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(
            test_loss.item(), test_micro_f1, test_macro_f1
        )
    )


if __name__ == "__main__":
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser("HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )
    parser.add_argument(
        "--hetero",
        action="store_true",
        help="Use metapath coalescing with DGL's own dataset",
    )
    parser.add_argument(
        "--dataset", type=str, default="ACM",
        help="dataset name"
    )
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help="weight decay")
    parser.add_argument("--num-epochs", type=int, default=200,
                        help="number of epochs")
    parser.add_argument('--num-heads', type=int, default=1,
                        help="number of heads")
    parser.add_argument("--hidden-units", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument('--dropout', type=float, default=0.6,
                        help="attention dropout")
    parser.add_argument("--patience", type=int, default=100,
                        help="early stopping patience value")

    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
