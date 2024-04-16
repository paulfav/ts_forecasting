import pandas as pd
import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from torch import nn
from sklearn.metrics import mean_squared_error
import random


use_cuda = torch.cuda.is_available()


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (c0, h0))
        out = self.fc(out[:, -1, :])
        return out

def train_model(x,y, batch_size = 64, hidden_size = 32, num_layers = 2, dropout_rate = 0.2, learning_rate = 0.001,input_size = 1, part_train = 0.7, part_valid = 0.2, epoch = 150): 
    random.seed(70)
    use_cuda = True

    # Data loading
    train_X = x[:int(part_train * len(x))]
    train_Y = y[:int(part_train * len(y))]
    valid_X = x[int(part_train * len(x)): int(part_train * len(x)) + int(part_valid * len(x))]
    valid_Y = y[int(part_train * len(y)): int(part_train * len(y)) + int(part_valid * len(y))]

    if input_size == 1:
        train_X, train_Y = torch.from_numpy(train_X).float().unsqueeze(-1), torch.from_numpy(train_Y).float()
        valid_X, valid_Y = torch.from_numpy(valid_X).float().unsqueeze(-1), torch.from_numpy(valid_Y).float()
    else:
        train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
        valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()

    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)    
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=batch_size)

    # Training parameters
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu") 
    model = LSTMModel(input_size, hidden_size, num_layers, dropout_rate=dropout_rate).to(device)      
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0

    train_loss_array_per_epoch = []
    valid_loss_array_per_epoch = []

    # Training epochs
    for epoch in tqdm(range(epoch)):
        model.train(True)                  
        train_loss_array = []
        hidden_train = None

        #Train phase
        for i, batch in enumerate(train_loader):
            batch_x, batch_y = batch[0].to(device), batch[1].to(device)
            pred_y = model(batch_x) 

            optimizer.zero_grad()               
            loss = criterion(pred_y, batch_y)  
            loss.backward()                     
            optimizer.step()     

            train_loss_array.append(loss.item())
            global_step += 1
                
        #Eval phase
        model.eval()                    
        valid_loss_array = []

        for i, batch in enumerate(valid_loader):
            batch_v_x , batch_v_y= batch[0].to(device), batch[1].to(device)
            pred_Y= model(batch_v_x)

            loss = criterion(pred_Y, batch_v_y)  

            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        train_loss_array_per_epoch.append(train_loss_cur)
        valid_loss_array_per_epoch.append(valid_loss_cur)
        
        # Save if better
        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(),"LSTMModel.pth")
        
        else:
            bad_epoch += 1
            if bad_epoch > 10:
                break
    print("Best valid loss: ", valid_loss_cur)
    print("Best train loss: ", train_loss_cur)

def test_model(x,y, hidden_size = 32, num_layers = 2, dropout_rate = 0.2, input_size = 1, part_train = 0.7, part_valid = 0.2): 


    test_X = x[int((part_train + part_valid) * len(x)):]
    true_Y = y[int((part_train + part_valid) * len(y)):]

    if input_size == 1:
        test_X = torch.from_numpy(test_X).float().unsqueeze(-1)
    else:
        test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size, hidden_size, num_layers, dropout_rate=dropout_rate).to(device)      
    model.load_state_dict(torch.load("LSTMModel.pth"))   

    result = torch.Tensor().to(device)

    model.eval()

    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X = model(data_X)
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    pred_Y = result.detach().cpu().numpy()
    
    return pred_Y, true_Y
    




