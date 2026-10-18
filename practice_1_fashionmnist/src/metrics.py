import torch
from torch import nn


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
    return dict(loss=total_loss / len(loader.dataset),
                accuracy=correct / len(loader.dataset), count=len(loader.dataset))


def load_checkpoint(path, device):
    from model import FashionMLP

    saved = torch.load(path, map_location=device, weights_only=True)
    model = FashionMLP(**saved['model_config'])
    model.load_state_dict(saved['model_state'])
    return model.to(device).eval(), saved


def classification_metrics(confusion):
    matrix = torch.as_tensor(confusion, dtype=torch.float64)
    support, predicted = matrix.sum(1), matrix.sum(0)
    tp = matrix.diag()
    fp, fn = predicted - tp, support - tp
    tn = matrix.sum() - tp - fp - fn
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    classes = [dict(precision=float(precision[i]), recall=float(recall[i]),
                    f1=float(f1[i]), support=int(support[i]),
                    TP=int(tp[i]), TN=int(tn[i]), FP=int(fp[i]), FN=int(fn[i]))
               for i in range(len(support))]
    return dict(classes=classes, macro_f1=float(f1.mean()),
                weighted_f1=float((f1 * support).sum() / support.sum().clamp_min(1)))
