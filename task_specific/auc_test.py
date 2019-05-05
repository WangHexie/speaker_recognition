import pandas as pd

sub = pd.read_csv("D:\\af2019-sr-devset-20190312\\submit.csv")
ano = pd.read_csv("D:\\af2019-sr-devset-20190312\\annotation.csv")
enrollment = pd.read_csv("D:\\af2019-sr-devset-20190312\\enrollment.csv")

data = pd.merge(sub, ano, on="FileID")
data = pd.merge(data, enrollment, on="SpeakerID")

Y = data[((data["IsMember_x"] == 'Y') & (data["IsMember_y"] == 'Y'))].count()
N = data[((data["IsMember_x"] == 'N') & (data["IsMember_y"] == 'N'))].count()

# NP1 = data[((data["IsMember_x"] == 'Y') & (data["IsMember_y"] == 'N') & (data["DeviceID"] == 1))].count()
# NP0 = data[((data["IsMember_x"] == 'Y') & (data["IsMember_y"] == 'N') & (data["DeviceID"] == 0))].count()
#
# NP1 = data[((data["IsMember_x"] == 'N') & (data["IsMember_y"] == 'Y') & (data["DeviceID"] == 1))].count()
# NP0 = data[((data["IsMember_x"] == 'N') & (data["IsMember_y"] == 'N') & (data["DeviceID"] == 0))].count()

print(Y)
print(N)
print((Y + N) / len(data))
