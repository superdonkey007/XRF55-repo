import os
import shutil

from tqdm import tqdm


def split_train_test_new(root_ns='/root/autodl-tmp/dataset/Raw_dataset/', dst_wr='/root/autodl-tmp/dataset/XRF_dataset/', split=14):
    dst_train_rfid = dst_wr + 'train_data/RFID/'
    dst_train_wifi = dst_wr + 'train_data/WiFi/'
    dst_train_mmwave = dst_wr + 'train_data/mmWave/'

    dst_test_rfid = dst_wr + 'test_data/RFID/'
    dst_test_wifi = dst_wr + 'test_data/WiFi/'
    dst_test_mmwave = dst_wr + 'test_data/mmWave/'

    if not os.path.exists(dst_train_rfid):
        os.makedirs(dst_train_rfid)
    if not os.path.exists(dst_train_wifi):
        os.makedirs(dst_train_wifi)
    if not os.path.exists(dst_train_mmwave):
        os.makedirs(dst_train_mmwave)
    if not os.path.exists(dst_test_rfid):
        os.makedirs(dst_test_rfid)
    if not os.path.exists(dst_test_wifi):
        os.makedirs(dst_test_wifi)
    if not os.path.exists(dst_test_mmwave):
        os.makedirs(dst_test_mmwave)

    for file in tqdm(os.listdir(root_ns + 'RFID/')):
        filename = file.split('.')[0]
        actidx = int(filename.split('_')[2])
        if actidx <= split:
            shutil.copy(root_ns + 'RFID/' + filename + '.npy', dst_train_rfid + filename + '.npy')
            shutil.copy(root_ns + 'WiFi/' + filename + '.npy', dst_train_wifi + filename + '.npy')
            shutil.copy(root_ns + 'mmWave/' + filename + '.npy', dst_train_mmwave + filename + '.npy')
        else:
            shutil.copy(root_ns + 'RFID/' + filename + '.npy', dst_test_rfid + filename + '.npy')
            shutil.copy(root_ns + 'WiFi/' + filename + '.npy', dst_test_wifi + filename + '.npy')
            shutil.copy(root_ns + 'mmWave/' + filename + '.npy', dst_test_mmwave + filename + '.npy')


if __name__ == '__main__':
    split_train_test_new(root_ns='/root/autodl-tmp/dataset/Raw_dataset/', dst_wr='/root/autodl-tmp/dataset/XRF_dataset/', split=14)
