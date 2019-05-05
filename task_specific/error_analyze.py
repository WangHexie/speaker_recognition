import pandas as pd

sub = pd.read_csv("D:\\af2019-sr-devset-20190312\\submit.csv")
ano = pd.read_csv("D:\\af2019-sr-devset-20190312\\annotation.csv")
enrollment = pd.read_csv("D:\\af2019-sr-devset-20190312\\enrollment.csv")

data = pd.merge(sub, ano, on="FileID")

True_same_id = 0
True_diff_id = 0
False_same_id = 0
False_diff_id = 0

for index, row in enrollment.iterrows():
    test_rows = data[data['SpeakerID'] == row.SpeakerID]
    for _, test_row in test_rows.iterrows():
        if test_row.IsMember_x == test_row.IsMember_y:
            if test_row.DeviceID == row.DeviceID:
                True_same_id += 1
            else:
                True_diff_id += 1
        else:
            if test_row.DeviceID == row.DeviceID:
                False_same_id += 1
            else:
                False_diff_id += 1


print(True_same_id/3, True_diff_id/3, False_same_id/3, False_diff_id/3)


