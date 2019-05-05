import argparse
import os

import numpy as np
import pandas as pd
from math import ceil

from dataset import DataSet
from feature_transform import distance
from feature_transform import get_mean_feature_for_device
from feature_transform import mean_vectors
from model import load_model
import keras.backend as K

parser = argparse.ArgumentParser("speaker recognition", fromfile_prefix_chars='@')
parser.add_argument('--file_dir', type=str, help='Directory of test data.')
parser.add_argument('--model_path', type=str, help='Directory to load model.')
parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help='sample rate of wave')
parser.add_argument('-s', '--output_shape', type=int, nargs=2, default=[32, 1024], help='shape')
parser.add_argument('-mt', '--model_type', type=int, default=2,
                    help='type of model.0:res_plus_transformer; 1.simple_cnn; 2.res_net')
args = parser.parse_args()
root_file = args.file_dir
data_path = os.path.join(root_file, "data1")
model_path = args.model_path
sample_rate = args.sample_rate
output_shape = args.output_shape
model_type = args.model_type
process_type = 1

def get_group_feature():
    model = load_model(model_path, load_type=model_type)
    data = pd.read_csv(os.path.join(root_file, "enrollment.csv"))
    dataset = DataSet(file_dir='', output_shape=output_shape, sample_rate=sample_rate)

    rows_list = []
    for name, group in data.groupby('GroupID', as_index=False):

        for person, file in group.groupby('SpeakerID'):
            li = []
            for i in file['FileID'].values:
                # print(model.predict(wav2mfcc(os.path.join(data_path, i+'.wav'))))
                arr = np.array(dataset.get_register_data(os.path.join(data_path, i + '.wav'), process_type))
                li.append(model.predict(arr.reshape((1, *arr.shape))))

            personLi = [person]
            personLi.extend((mean_vectors(li)[0]))
            groupLi = [name]
            groupLi.extend(personLi)
            rows_list.append(groupLi)
    res = pd.DataFrame(rows_list)
    # print(res)
    res.to_csv(os.path.join(root_file, 'enroll.csv'))
    return rows_list


def save_test():
    """
    保存测试集
    :return:
    """
    return


def test_output(is_handle_device=False):
    model = load_model(model_path, load_type=model_type)
    data = pd.read_csv(os.path.join(root_file, "test.csv"))
    dataset = DataSet(file_dir='', output_shape=output_shape, sample_rate=sample_rate)
    # device_arr = get_mean_feature_for_device(model_path=model_path, path=root_file,
    #                                          output_shape=output_shape, sample_rate=sample_rate, process_class=process_type)

    for group, attack in data.groupby("DeviceID"):

        cur_device_type_arr = []
        fileLi = attack["FileID"].values
        groupID = attack["GroupID"].values

        for i in fileLi:
            wav_data = dataset.get_test_data(os.path.join(data_path, i + ".wav"), process_type)
            # print(np.array(wav_data).shape)
            cur_device_type_arr.append(wav_data)
        arr = np.array(cur_device_type_arr)

        p = []

        num = 5
        time = ceil(len(arr) / num)
        for i in range(time):
            p.append(arr[num * i: num * (i + 1)])

        K.clear_session()
        del model
        model = load_model(model_path, model_type)
        result = []
        for i in p:
            result.append(model.predict(np.array(i)))

        model_predict_data = np.concatenate(tuple(result))


        # model_predict_data = model.predict(arr)
        if is_handle_device:
            if group == 1:
                diff = device_arr[0] - device_arr[1]
                diff = diff.reshape(1, diff.shape[0])
                model_predict_data = model_predict_data + diff.repeat(model_predict_data.shape[0], axis=0)

        arr = np.r_[fileLi.reshape(1, fileLi.shape[0]), groupID.reshape(1, groupID.shape[0])]
        arr = arr.T
        arr = np.c_[arr, model_predict_data]

        if group == 1:
            pd.DataFrame(arr).to_csv(os.path.join(root_file, "attack_device_1.csv"))
        if group == 0:
            pd.DataFrame(arr).to_csv(os.path.join(root_file, "attack_device_0.csv"))


def get_min_distance():
    enroll = pd.read_csv(os.path.join(root_file, "enroll.csv"), index_col=0)
    attacker = pd.read_csv(os.path.join(root_file, "attack_feature.csv"), index_col=0)
    # print(attacker.columns)

    res = pd.DataFrame()
    for name, group in attacker.groupby("1"):
        li = []
        fileLi = group["0"].values
        groupID = group["1"].values
        for i in group.values:
            attacker_feature = i[2:]
            # print(len(attacker_feature))
            group_id = i[1]
            file_id = i[0]
            # print(group_id , file_id)

            group_features = np.array(enroll[enroll["0"] == group_id].loc[:, "2":])
            attacker_feature = attacker_feature.astype("float64")
            dis = distance(attacker_feature, group_features)
            li.append(min(dis))

        arr = np.r_[fileLi.reshape(1, fileLi.shape[0]), groupID.reshape(1, groupID.shape[0])]
        # print(arr.shape)
        arr = arr.T
        arr = np.c_[arr, li]
        df = pd.DataFrame(arr)
        df.columns = ['fileID', 'group', 'dis']
        df["is_member"] = "N"
        df["dis"] = df["dis"].astype("float64")
        # mean = np.mean(df["dis"])
        # print(mean)
        # df.loc[df["dis"] < mean, "is_member"] = "Y"
        top50 = np.array(df["dis"].values)
        top50 = np.sort(top50, axis=0)[50]
        # print(df[df["dis"] <= top50].loc[:,"is_member"])
        df.loc[df["dis"] <= top50, "is_member"] = "Y"
        print(df)

        res = pd.concat([res, df], axis=0)
    res = res.loc[:, ["fileID", "is_member"]]
    # print(res)
    res.columns = ["FileID", "IsMember"]
    res.to_csv(os.path.join(root_file, "submit.csv"), index=False)


if __name__ == '__main__':
    get_group_feature()
    test_output(is_handle_device=False)
    attacker_1 = pd.read_csv(os.path.join(root_file, "attack_device_1.csv"), index_col=0)
    attacker_0 = pd.read_csv(os.path.join(root_file, "attack_device_0.csv"), index_col=0)
    pd.concat([attacker_0, attacker_1], axis=0).to_csv(os.path.join(root_file, "attack_feature.csv"))
    get_min_distance()
    # enroll = pd.read_csv(os.path.join(root_file, "enroll.csv") , index_col=0)
    # test_output()
    # device_arr = get_mean_feature_for_device(model_path=model_path, path=root_file,
    #                                          output_shape=output_shape, sample_rate=sample_rate)
