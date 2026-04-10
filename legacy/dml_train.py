import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import datetime

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from tqdm import tqdm

import XRFDataset
from opts import parse_opts
from legacy.model import resnet1d, resnet1d_rfid, resnet2d


if __name__ == '__main__':
    model_name = 'train_dml'
    scene = model_name.split('_')[-1]
    print(model_name)
    starttime = datetime.datetime.now()
    args = parse_opts()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_pin_memory = device.type == 'cuda'

    train_dataset = XRFDataset.XRFBertDatasetNewMix(is_train=True, scene=scene)
    train_size = len(train_dataset)
    train_data = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=4,
        drop_last=False,
    )

    models = [
        resnet1d.resnet18_mutual().to(device),
        resnet1d_rfid.resnet18_mutual().to(device),
        resnet2d.resnet18_mutual().to(device),
    ]

    optimizers = []
    schedulers = []
    for m in range(args.model_num):
        optimizers.append(torch.optim.Adam(models[m].parameters(), lr=args.lr))
        schedulers.append(
            torch.optim.lr_scheduler.MultiStepLR(
                optimizers[m],
                milestones=[40, 80, 120, 160],
                gamma=0.5,
            )
        )

    train_loss = np.zeros([args.model_num, args.epoch])
    print('--------------Mutual Learning---------------')
    print('Using device: {}'.format(device))
    print('Train on {} samples.'.format(train_size))

    loss_ce = nn.CrossEntropyLoss().to(device)
    loss_l1 = torch.nn.L1Loss().to(device)
    loss_diff = nn.KLDivLoss(reduction='batchmean').to(device)

    idx = 0
    save_dir = '/root/autodl-tmp/xrf_result/params/' + model_name + '/'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epoch):
        print('Epoch:', epoch)
        losses = []
        for m in range(args.model_num):
            models[m].train()
            losses.append(0)

        for samples_wifi, samples_rfid, samples_mmwave, labels, vectors in tqdm(train_data):
            samples_wifi = samples_wifi.to(device)
            samples_rfid = samples_rfid.to(device)
            samples_mmwave = samples_mmwave.to(device)
            labels = labels.to(device)
            vectors = vectors.to(device)

            model_outputs = [0 for _ in range(args.model_num)]
            model_vecs = [0 for _ in range(args.model_num)]

            model_outputs[0], model_vecs[0] = models[0](samples_wifi)
            model_outputs[1], model_vecs[1] = models[1](samples_rfid)
            model_outputs[2], model_vecs[2] = models[2](samples_mmwave)

            loss = []
            for i in range(args.model_num):
                ce_loss = loss_ce(model_outputs[i], labels)
                vec_ce_loss = loss_l1(model_vecs[i], vectors)
                diff_loss = 0
                for j in range(args.model_num):
                    if i != j:
                        diff_loss += loss_diff(
                            F.log_softmax(model_outputs[i], dim=1),
                            F.softmax(model_outputs[j].detach(), dim=1),
                        )

                loss.append(vec_ce_loss + ce_loss + diff_loss / (args.model_num - 1))
                losses[i] += loss[i].item()

            for i in range(args.model_num):
                optimizers[i].zero_grad()
            for i in range(args.model_num):
                loss[i].backward(retain_graph=True)
            for i in range(args.model_num):
                optimizers[i].step()

        for i in range(args.model_num):
            schedulers[i].step()

        for m in range(args.model_num):
            train_loss[m, epoch] = losses[m] / train_size
            print('Model ', m, ' train_loss: ', losses[m] / train_size)
        if epoch >= args.epoch - 10:
            for m in range(args.model_num):
                torch.save(models[m].state_dict(), save_dir + f'model{m}_epoch{idx}.pth')
            idx += 1

    learning_curve_dir = '/root/autodl-tmp/xrf_result/learning_curve/' + model_name + '/'
    os.makedirs(learning_curve_dir, exist_ok=True)
    for m in range(args.model_num):
        sio.savemat(
            learning_curve_dir + model_name + '_' + str(m) + '_train_loss.mat',
            {'train_loss': train_loss[m]},
        )

    print(model_name)
    endtime = datetime.datetime.now()
    print(starttime)
    print(endtime)
    print((endtime - starttime).seconds)
