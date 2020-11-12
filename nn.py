import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(110)

class Net(nn.Module):

    def __init__(self, uin_dim, iin_dim, out_dim, use_similarity = True):
        super(Net, self).__init__()

        self.uin_dim = uin_dim
        self.iin_dim = iin_dim

        self.use_similarity = use_similarity

        self.ufc1 = nn.Linear(uin_dim, 4)
        torch.nn.init.xavier_uniform_(self.ufc1.weight)

        self.ifc1 = nn.Linear(iin_dim, 8)
        torch.nn.init.xavier_uniform_(self.ifc1.weight)
        self.ifc2 = nn.Linear(8, 4)
        torch.nn.init.xavier_uniform_(self.ifc2.weight)

        if use_similarity:
            self.fc1 = nn.Linear(9, 4)
        else:
            self.fc1 = nn.Linear(8, 4)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(4, out_dim)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.Cos = nn.CosineSimilarity()

    def forward(self, x):

        if self.use_similarity:
            user_pref = x[:, 2:12]
            anime_pref = x[:, 22:32]

            Sim = self.Cos(user_pref, anime_pref)
            Sim = Sim * 10
            Sim.resize_((Sim.shape[0],1))

        user_x = x[:, :self.uin_dim]
        item_x = x[:, self.uin_dim:]

        u1 = F.relu(self.ufc1(user_x))

        i1 = F.relu(self.ifc1(item_x))
        i2 = F.relu(self.ifc2(i1))

        in_x = torch.cat((u1, i2), dim=1)
        if self.use_similarity:
            in_x = torch.cat((in_x, Sim), dim=1)

        out = F.relu(self.fc1(in_x))

        out = self.fc2(out)

        return out

def load_data(train_file, val_file, test_file):
    categorical_columns = ['Gender', 'Rating', 'Producers', 'Studios', 'Type']
    numerical_columns = ['Episodes Watched', 'Episodes', 'Average Score', 'Popularity', 'Members', 'Favorites', 'Adaption Size']

    train_dataframe = pd.read_csv(train_file)
    test_dataframe = pd.read_csv(test_file)
    val_dataframe = pd.read_csv(val_file)

    train_dataframe.drop(columns=['ID'], axis=1, inplace=True)
    test_dataframe.drop(columns=['ID'], axis=1, inplace=True)
    val_dataframe.drop(columns=['ID'], axis=1, inplace=True)

    for category in categorical_columns:
        train_dataframe[category] = (train_dataframe[category].astype('category')).cat.codes.values
        test_dataframe[category] = (test_dataframe[category].astype('category')).cat.codes.values
        val_dataframe[category] = (val_dataframe[category].astype('category')).cat.codes.values

    scaler = MinMaxScaler()
    transform_train_data = scaler.fit_transform(train_dataframe[numerical_columns])
    transform_test_data = scaler.transform(test_dataframe[numerical_columns])
    transform_val_data = scaler.transform(val_dataframe[numerical_columns])

    for i, numeric in enumerate(numerical_columns):
        train_dataframe[numeric] = transform_train_data[:, i]
        test_dataframe[numeric] = transform_test_data[:, i]
        val_dataframe[numeric] = transform_val_data[:, i]

    train_x = torch.from_numpy(np.array(train_dataframe)[:, :-1])
    train_y = torch.from_numpy((np.array(train_dataframe)[:, -1]).reshape(-1, 1))

    test_x = torch.from_numpy(np.array(test_dataframe)[:, :-1])
    test_y = torch.from_numpy((np.array(test_dataframe)[:, -1]).reshape(-1, 1))

    val_x = torch.from_numpy(np.array(val_dataframe)[:, :-1])
    val_y = torch.from_numpy((np.array(val_dataframe)[:, -1]).reshape(-1, 1))

    return train_x,  train_y, test_x, test_y, val_x, val_y

def train_and_predict(model, iter, train_x, train_y, predict_x, predict_y):
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for t in range(iter):
        y_pred = model(train_x.float())
        loss = criterion(y_pred, train_y.float())
        if t % 1000 == 0:
            print(t, math.sqrt(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    y_pred = model(train_x.float())
    calculate_accuracy(y_pred, train_y)
    loss = criterion(y_pred, train_y.float())
    print("final train loss is " + str(math.sqrt(loss.item())))

    y_pred = model(predict_x.float())
    torch.round(y_pred)
    loss = criterion(y_pred, predict_y.float())
    return (y_pred,loss)

def calculate_accuracy(pred, truth):
    pred = pred.detach().numpy()
    truth = truth.detach().numpy()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    threshold = 7
    for i in range(pred.shape[0]):
        if pred[i][0] >= threshold and truth[i][0] >= threshold:
            TP += 1
        elif pred[i][0] < threshold and truth[i][0] < threshold:
            TN += 1
        elif pred[i][0] < threshold and truth[i][0] >= threshold:
            FN += 1
        elif pred[i][0] >= threshold and truth[i][0] < threshold:
            FP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    print("precision " + str(precision))
    print("recall " + str(recall))
    print("F1 " + str(F1))

def NN_with_similarity(train_x, train_y, test_x, test_y, val_x, val_y, find_best=False):
    best_iteration = 16000
    min_loss = 10000

    find_best_iteration = find_best
    if find_best_iteration:
        for i in range(10):
            iter = i*1000 + 10000
            model = Net(12, 20, 1)
            y_pred, loss = train_and_predict(model, iter, train_x, train_y, val_x, val_y)
            print("for iteration " + str(iter) + " RMSE is " + str(math.sqrt(loss.item())))
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_iteration = iter

    print("min validation loss is " + str(min_loss))
    print("best iteration is " + str(best_iteration))

    test_model = Net(12, 20, 1)
    y_pred, loss = train_and_predict(test_model, best_iteration, train_x, train_y, test_x, test_y)
    print("test RMSE is " + str(math.sqrt(loss.item())))

    calculate_accuracy(y_pred, test_y)

def NN_with_no_similarity(train_x, train_y, test_x, test_y, val_x, val_y, find_best=False):
    best_iteration = 16000
    min_loss = 10000

    find_best_iteration = find_best
    if find_best_iteration:
        for i in range(10):
            iter = i*1000 + 10000
            model = Net(12, 20, 1, False)
            y_pred, loss = train_and_predict(model, iter, train_x, train_y, val_x, val_y)
            print("for iteration " + str(iter) + " RMSE is " + str(math.sqrt(loss.item())))
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_iteration = iter

    print("min validation loss is " + str(min_loss))
    print("best iteration is " + str(best_iteration))

    test_model = Net(12, 20, 1, False)
    y_pred, loss = train_and_predict(test_model, best_iteration, train_x, train_y, test_x, test_y)
    print("test RMSE is " + str(math.sqrt(loss.item())))

    calculate_accuracy(y_pred, test_y)

train_x, train_y, test_x, test_y, val_x, val_y = load_data("train_data.csv", "val_data.csv", "test_data.csv")
#NN_with_similarity(train_x, train_y, test_x, test_y, val_x, val_y, find_best=False)
NN_with_no_similarity(train_x, train_y, test_x, test_y, val_x, val_y, find_best=False)