import numpy as np
from keras.utils import Sequence, to_categorical
from sklearn.utils import shuffle


class VoiceSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, num_class, data_process_func):
        self.x, self.y = shuffle(x_set, to_categorical(y_set, num_classes=num_class))
        self.batch_size = batch_size
        self.data_process_function = data_process_func
        self.length = self.__len__()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(self.data_process_function(batch_x)), np.array(batch_y)

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)


class CenterLossSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, num_class, data_process_func):
        self.x, self.y = shuffle(x_set, y_set)
        self.batch_size = batch_size
        self.data_process_function = data_process_func
        self.num_of_class = num_class
        self.length = self.__len__()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size))) - 1

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return [np.array(self.data_process_function(batch_x)), np.array(batch_y).flatten()], [to_categorical(np.array(batch_y), num_classes=self.num_of_class), np.zeros(self.batch_size)]

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)


class SiameseSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, num_class, data_process_func):
        self.x = [[] for _ in range(len(set(y_set)))]
        for i in range(len(x_set)):
            self.x[y_set[i]].append(x_set[i])
        self.batch_size = batch_size
        self.data_process_function = data_process_func
        self.length = self.__len__()
        self.num_class = num_class
        self.iter = 0

    def __len__(self):
        return int(np.ceil(len(self.x) * 2 / float(self.batch_size) - 1))

    def __getitem__(self, idx):
        left = []
        right = []
        label = []
        half_of_batch_size = int(self.batch_size / 2)
        for i in range(half_of_batch_size):
            left_index_now = (idx * half_of_batch_size + i) % len(self.x)

            # add different class pair
            k = np.random.randint(len(self.x[left_index_now]))
            left.append(self.x[left_index_now][k])
            different_cls = (np.random.randint(len(self.x) - 1) + left_index_now) % len(self.x)
            right.append(self.x[different_cls][np.random.randint(len(self.x[different_cls]))])

            # add same class pair
            k = np.random.randint(len(self.x[idx * half_of_batch_size + i]))
            left.append(self.x[left_index_now][k])
            different_idx = (np.random.randint(len(self.x[left_index_now]) - 1) + k) % len(self.x[left_index_now])
            right.append(self.x[left_index_now][different_idx])

            # add label
            label += [0, 1]

        left, right, label = shuffle(left, right, label)

        return [np.array(self.data_process_function(left)), np.array(self.data_process_function(right))], label


class TripletSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, num_class, data_process_func):
        self.x = [[] for _ in range(len(set(y_set)))]
        for i in range(len(x_set)):
            self.x[y_set[i]].append(x_set[i])
        self.batch_size = batch_size
        self.data_process_function = data_process_func
        self.length = self.__len__()
        self.num_class = num_class
        self.iter = 0

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    # def on_epoch_end(self):
    #     for i in range(len(self.x)):
    #         self.x[i] = shuffle(self.x[i])

    def __getitem__(self, idx):
        user_input = []
        positive_input = []
        negative_input = []
        label = []
        if idx == 0:
            self.iter += 1

        if (self.iter % 520) == 0:
            for i in range(len(self.x)):
                self.x[i] = shuffle(self.x[i])

        for i in range(self.batch_size):
            user_class_index_now = (idx * self.batch_size + i) % len(self.x)

            # add different class pair
            data_index = self.iter % len(self.x[user_class_index_now])
            user_input.append(self.x[user_class_index_now][data_index])

            different_cls = (np.random.randint(len(self.x) - 1) + user_class_index_now) % len(self.x)
            negative_input.append(self.x[different_cls][np.random.randint(len(self.x[different_cls]))])

            # add same class pair
            same_class_different_index = (np.random.randint(
                len(self.x[user_class_index_now]) - 1) + data_index) % len(self.x[user_class_index_now])
            positive_input.append(self.x[user_class_index_now][same_class_different_index])
            label += [1]

        return [np.array(self.data_process_function(positive_input)),
                np.array(self.data_process_function(negative_input)),
                np.array(self.data_process_function(user_input))], np.array(label)
