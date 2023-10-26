import os

import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import *
import vars as v
import json

def train():
    v.model.train()
    t = tqdm(total=len(v.trainloader), desc=f'Train epoch: {v.current_epoch}')
    for batch_idx, (data, target) in enumerate(v.trainloader):
        data, target = data.to(args.device), target.to(args.device)
        v.optimizer.zero_grad()
        output = v.model(data)
        loss = v.criterion(output, target)
        loss.backward()
        v.optimizer.step()
        t.update(1)
        t.set_postfix({'Loss': {loss.item()}})
    t.close()

def test():
    v.model.eval()
    labels = []
    preds = []
    t = tqdm(total=len(v.validloader), desc=f'Test epoch: {v.current_epoch}')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(v.validloader):
            data, target = data.to(args.device), target.to(args.device)
            output = v.model(data)
            predicted = torch.argmax(output, dim=1)
            labels.extend(target.tolist())
            preds.extend(predicted.tolist())
            t.update(1)
    accuracy: float = accuracy_score(labels, preds)
    precision: float = precision_score(labels, preds, average="macro")
    recall: float = recall_score(labels, preds, average="macro")
    f1: float = f1_score(labels, preds, average="macro")
    t.close()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def loop():
    v.model = v.model.to(args.device)
    v.optimizer = optim.Adam(v.model.parameters(), lr=args.learning_rate)
    v.current_epoch = 1
    v.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    os.makedirs(f"{args.save_path}/{tag}", exist_ok=True)
    v.writer = SummaryWriter(log_dir=f"{args.save_path}/{tag}")
    with open(f"{args.save_path}/{tag}/{tag}.json", "w+") as f:
        vs = vars(args)
        json.dump(
            {k: vs[k] for k in vs if not k.startswith('_') and not hasattr(vs[k], "__dict__")},
            f,
            indent=2,
            default=lambda o: str(o),
        )
    
    while v.current_epoch <= args.num_epochs:
        train()
        for key, value in test().items():
            v.writer.add_scalar(key, value, v.current_epoch)
        torch.save(
            {
                "epoch": v.current_epoch,
                "model_state_dict": v.model.state_dict(),
                "optimizer_state_dict": v.optimizer.state_dict(),
            },
            f"{args.save_path}/{tag}/{tag}.pt",
        )
        v.current_epoch += 1

if __name__ == "__main__":
    globals()[config]()
    loop()
