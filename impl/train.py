import torch


def train(optimizer, model, dataloader, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        pred = model(*batch[:-1], id=0)
        loss = loss_fn(pred, batch[-1])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    return sum(total_loss) / len(total_loss) #if len(total_loss) else 0


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    for batch in dataloader:
        pred = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)

@torch.no_grad()
def test_full_report(model, dataloader, report_fn, loss_fn):
    """
    Final test report.
    Collects predictions and labels, computes loss, and prints full AML metrics.
    """
    model.eval()
    preds = []
    ys = []

    for batch in dataloader:
        pred = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])

    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)

    loss = loss_fn(pred, y)

    return report_fn(
        pred.cpu().numpy(),
        y.cpu().numpy(),
        loss=loss.detach().cpu().item(),
    )