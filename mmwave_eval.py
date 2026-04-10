import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import gdown
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import torch
import torch.utils.data as Data
from tqdm import tqdm

from mmwave_dataset import XRFMmwaveDataset
from model import resnet2d
from opts import parse_opts


def get_conf_matrix(pred, truth, conf_matrix):
    p = pred.tolist()
    l = truth.tolist()
    for i in range(len(p)):
        conf_matrix[l[i]][p[i]] += 1
    return conf_matrix


def write_to_file(conf_matrix, path):
    conf_matrix_m = conf_matrix
    for x in range(len(conf_matrix_m)):
        base = sum(conf_matrix_m[x])
        if base == 0:
            continue
        for y in range(len(conf_matrix_m[0])):
            conf_matrix_m[x][y] = format(conf_matrix_m[x][y] / base, '.2f')
    df = pd.DataFrame(conf_matrix_m)
    df.to_csv(path + '.csv', index=False)


def check_weight_file(path):
    url_mmwave = 'https://drive.google.com/file/d/1vJou1BnwCreCkqu-10mdg45tqARdGeU5/view?usp=sharing'
    mmwave_weight = os.path.join(path, 'model_epoch6.pth')

    if not os.path.exists(mmwave_weight):
        print('downloading mmwave weights file.')
        gdown.download(url_mmwave, mmwave_weight, quiet=False, fuzzy=True)
    else:
        print('mmwave weights file already exists.')


if __name__ == '__main__':
    model_name = 'mmwave_eval_dml'
    scene = 'dml'
    print(model_name)
    args = parse_opts()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_pin_memory = device.type == 'cuda'

    weight_file_path = '/root/autodl-tmp/mmwave_result/params/'
    check_weight_file(weight_file_path)

    starttime = datetime.datetime.now()

    test_dataset = XRFMmwaveDataset(is_train=False, scene=scene)
    test_size = len(test_dataset)
    test_data = Data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=16,
    )

    model = resnet2d.resnet18_mutual()
    model.load_state_dict(torch.load('/root/autodl-tmp/mmwave_result/params/model_epoch6.pth', map_location=device))
    model = model.to(device)
    model.eval()

    print('--------------MMWave Only Eval---------------')
    print('Using device: {}'.format(device))
    print('validate on {} samples.'.format(test_size))

    test_loss = 0
    test_acc = 0
    max_acc = 0
    temp_test = 0
    correct_test = 0
    conf_matrix = [[0 for _ in range(args.class_num)] for _ in range(args.class_num)]

    for samples_mmwave, labels, vectors in tqdm(test_data):
        with torch.no_grad():
            samples_mmwave = samples_mmwave.to(device)
            labels_device = labels.to(device)

            model_outputs, _ = model(samples_mmwave)
            prediction = model_outputs.data.max(1)[1]
            correct_test += prediction.eq(labels_device.data).sum().item()
            conf_matrix = get_conf_matrix(prediction.cpu(), labels, conf_matrix)

    acc = 100 * float(correct_test) / test_size
    print('MMWave Model Test accuracy:', acc)

    max_acc = max(max_acc, acc)
    test_acc = acc
    testacc_str = str(acc)[0:6]

    if correct_test > temp_test:
        plt.matshow(conf_matrix, cmap=plt.cm.Reds)
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix)):
                plt.text(
                    j,
                    i,
                    str(conf_matrix[i][j]),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=4,
                )
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        conf_dir = '/root/autodl-tmp/mmwave_result/conf_matrix/' + model_name + '/2/'
        weight_dir = '/root/autodl-tmp/mmwave_result/weights/' + model_name + '/2/'

        os.makedirs(conf_dir, exist_ok=True)
        os.makedirs(weight_dir, exist_ok=True)

        plt.savefig(conf_dir + model_name + '_' + testacc_str + '.jpg', dpi=300)
        write_to_file(conf_matrix, weight_dir + model_name + '_' + testacc_str)
        torch.save(model, weight_dir + model_name + '_' + testacc_str + '.pkl')
        plt.close()
        temp_test = correct_test

    print('max_accuracy:', max_acc)

    learning_curve_dir = '/root/autodl-tmp/mmwave_result/learning_curve/' + model_name + '/'
    os.makedirs(learning_curve_dir, exist_ok=True)

    sio.savemat(learning_curve_dir + model_name + '_2_test_loss.mat', {'test_loss': test_loss})
    sio.savemat(learning_curve_dir + model_name + '_2_test_acc.mat', {'test_acc': test_acc})

    print(model_name)
    endtime = datetime.datetime.now()
    print(starttime)
    print(endtime)
    print((endtime - starttime).seconds)
