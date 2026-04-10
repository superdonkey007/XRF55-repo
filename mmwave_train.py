import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm

from mmwave_dataset import XRFMmwaveDataset
from model import resnet2d
from opts import parse_opts


if __name__ == '__main__':
    model_name = 'mmwave_dml'
    scene = 'dml'
    print(model_name)

    starttime = datetime.datetime.now()
    args = parse_opts()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_pin_memory = device.type == 'cuda'

    train_dataset = XRFMmwaveDataset(is_train=True, scene=scene)
    train_size = len(train_dataset)
    train_data = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=4,
        drop_last=False,
    )

    model = resnet2d.resnet18_mutual().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[40, 80, 120, 160],
        gamma=0.5,
    )

    train_loss = np.zeros([args.epoch])

    print('--------------Train MmWave Only---------------')
    print('Using device: {}'.format(device))
    print('Train on {} samples.'.format(train_size))

    loss_ce = nn.CrossEntropyLoss().to(device)
    loss_l1 = nn.L1Loss().to(device)

    save_path = '/root/autodl-tmp/mmwave_result/'
    os.makedirs(save_path + 'params/', exist_ok=True)

    idx = 0

    for epoch in range(args.epoch):
        print('Epoch:', epoch)
        model.train()
        total_loss = 0

        for samples_mmwave, labels, vectors in tqdm(train_data):
            samples_mmwave = samples_mmwave.to(device)
            labels = labels.to(device)
            vectors = vectors.to(device)

            outputs, vecs = model(samples_mmwave)
            ce_loss = loss_ce(outputs, labels)
            vec_loss = loss_l1(vecs, vectors)
            loss = ce_loss + vec_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        train_loss[epoch] = total_loss / train_size
        print('Train loss:', train_loss[epoch])

        if epoch >= args.epoch - 10:
            torch.save(model.state_dict(), f'{save_path}params/model_epoch{idx}.pth')
            idx += 1

    os.makedirs(save_path + 'learning_curve/', exist_ok=True)
    sio.savemat(save_path + 'learning_curve/train_loss.mat', {'train_loss': train_loss})

    print(model_name)
    endtime = datetime.datetime.now()
    print(starttime)
    print(endtime)
    print((endtime - starttime).seconds)
