import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import *
import models
from utils import visualization
from utils import metrics
import vars as v
import json

def load_checkpoint(filepath = None):
    if filepath == None:
        raise Exception('Specify filepath')
    else:
        checkpoint = torch.load(filepath)
        v.optimizer = optim.SGD(v.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        v.model.load_state_dict(checkpoint['model_state_dict'])
        v.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        v.current_epoch = checkpoint['epoch']

def train():
    v.model.train()
    train_loss = 0
    t = tqdm(total=len(v.trainloader), desc=f'Train epoch: {v.current_epoch}')

    for batch_idx, (data, target) in enumerate(v.trainloader):
        data, target = data.to(args.device), target.to(args.device)

        v.optimizer.zero_grad()
        output = v.model(data)
        loss = v.criterion(output, target)
        train_loss += loss
        loss.backward()
        v.optimizer.step()

        t.update(1)
        t.set_postfix({'Loss': {loss.item()}})

    t.close()
    train_loss /= len(v.trainloader.dataset)

    return 'loss/train', train_loss

def test():
    v.model.eval()
    labels = []
    preds = []
    valid_loss = 0
    t = tqdm(total=len(v.validloader), desc=f'Test epoch: {v.current_epoch}')

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(v.validloader):
            data, target = data.to(args.device), target.to(args.device)
            output = v.model(data)
            loss = v.criterion(output, target)
            predicted = torch.argmax(output, dim=1)
            valid_loss += loss
            labels.extend(target.tolist())
            preds.extend(predicted.tolist())
            t.update(1)

    valid_loss /= len(v.validloader.dataset)
    labels, preds = torch.tensor(labels), torch.tensor(preds)

    # calculating metrics
    global_accuracy: float = metrics.global_accuracy_score(labels, preds)
    class_accuracy: float = metrics.class_average_accuracy_score(labels, preds, v.num_classes)
    mean_IOU: float = metrics.mean_IOU(labels, preds, v.num_classes)
    fw_IOU: float = metrics.freq_weighted_IOU(labels, preds, v.num_classes)
    precision: float = metrics.precision_score(labels, preds, v.num_classes)
    recall: float = metrics.recall_score(labels, preds, v.num_classes)
    f1: float = metrics.f1_score(labels, preds, v.num_classes)

    t.close()
    # compare output with groundtruth mask
    sample = next(iter(v.validloader))
    data, mask = sample[0][0].unsqueeze(0).to(args.device), sample[1][0].to(args.device)
    visualization.visualize(data, torch.argmax(v.model(data), dim=1).squeeze(0), mask)

    return {
        "loss/validation": valid_loss,
        "global accuracy": global_accuracy,
        "class avg accuracy": class_accuracy,
        "mean IOU": mean_IOU,
        "frequency weighted IOU": fw_IOU,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def loop():
    # initalize variables
    if not dict_path:
        v.optimizer = optim.SGD(v.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        v.current_epoch = 1

    v.lr_scheduler = ReduceLROnPlateau(v.optimizer, mode='min', factor=0.1)
    v.model = v.model.to(args.device)
    v.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Write the hyperparameters into a json file
    os.makedirs(f"{args.save_path}/{tag}", exist_ok=True)
    with open(f"{args.save_path}/{tag}/{tag}.json", "w+") as f:
        vs = vars(args)
        json.dump(
            {k: vs[k] for k in vs if not k.startswith('_') and not hasattr(vs[k], "__dict__")},
            f,
            indent=2,
            default=lambda o: str(o),
        )
    
    v.writer = SummaryWriter(log_dir=f"{args.save_path}/{tag}")
    while v.current_epoch <= args.num_epochs:
        key, value = train()
        v.writer.add_scalar(key, value, v.current_epoch)

        # writes the metric results into tensorboard
        for key, value in test().items():
            v.writer.add_scalar(key, value, v.current_epoch)

        # saves model state
        torch.save(
            {
                "epoch": v.current_epoch,
                "model_state_dict": v.model.state_dict(),
                "optimizer_state_dict": v.optimizer.state_dict(),
            },
            f"{args.save_path}/{tag}/{tag}.pt",
        )

        v.lr_scheduler.step()
        v.current_epoch += 1

if __name__ == "__main__":
    globals()[dataset_name]()
    v.model = getattr(models, model_name)()
    if dict_path:
        load_checkpoint(dict_path)
    loop()
