from LSTMmodel import *



part_train =  len(pd.read_csv('train_normalised.csv')) / (len(pd.read_csv('train_normalised.csv')) + len(pd.read_csv('test_normalised.csv')))
part_valid = 0.2 * part_train
part_train =  0.8 * part_train

def getfeatures(input_size = 1, timeWindowToUse = 15):
    data_train = pd.read_csv('train_normalised.csv')
    data_train.date = pd.to_datetime(data_train.date)
    data_test = pd.read_csv('test_normalised.csv')
    data_test.date = pd.to_datetime(data_test.date)

    part_train =  len(data_train) / (len(data_train) + len(data_test))

    part_valid = 0.2 * part_train
    part_train =  0.8 * part_train
    stock_name = 'AAPL'

    if input_size == 1:
        df_series = pd.concat([data_train["close_" + stock_name], data_test["close_" + stock_name]])

        df = df_series.to_frame("y").astype(float)
        for i in range(timeWindowToUse, 0, -1):
            df["x_{}".format(i)] = df["y"].shift(i)
        df.dropna(inplace=True)

        y = df["y"].to_numpy().reshape(-1,1)
        x = df.drop(columns=["y"]).to_numpy()
        return x, y
    
    else :
        df = pd.concat([data_train[data_train.columns[4::6]], data_test[data_test.columns[4::6]]])
        x = np.array([df.shift(i).to_numpy() for i in range(timeWindowToUse, 0, -1)])

        x = np.concatenate(x, axis=1).reshape(-1, timeWindowToUse, 8)
        y = df["close_" + stock_name].to_numpy().reshape(-1,1)
        return x[timeWindowToUse:], y[timeWindowToUse:]

    
def run( batch_size = 16, hidden_size = 32, num_layers = 1, dropout_rate = 0, input_size = 1, epoch = 5, timeWindowToUse = 15, learning_rate = 0.005):
    if input_size == 1:
        x, y  = getfeatures(1, timeWindowToUse)
        input_size = 1
    else :
        x, y  = getfeatures(8, timeWindowToUse)
        input_size = 8

    train_model(x, y, batch_size = batch_size, hidden_size = hidden_size, num_layers = num_layers,
                 dropout_rate = dropout_rate, input_size = input_size, part_train = part_train, part_valid = part_valid,
                  learning_rate= learning_rate, epoch = epoch)
    pred_Y, true_Y = test_model(x, y, hidden_size = hidden_size, num_layers = num_layers, dropout_rate = dropout_rate, input_size = input_size, part_train = part_train, part_valid = part_valid)
    
    mse = mean_squared_error(true_Y, pred_Y)

    res = pd.read_csv('results.csv')
    dicToAdd = {'Model': 'LSTM','input_size': input_size,'Window': timeWindowToUse,'layers': num_layers,'hidden': hidden_size,
                    'batch_size' : batch_size, 'dropout': dropout_rate, 'learning_rate': learning_rate,
                    'epochs': epoch, 'mse': mse}
    res = pd.concat([res, pd.DataFrame.from_records([dicToAdd])])
    print(dicToAdd)
    print("MSE: ", mse) 
    res.to_csv('results.csv', index=False)



bcs = [8, 16, 32, 64, 128]
hss = [8, 16, 32, 64, 128]
nls = [1, 2, 3, 4]
drs = [0, 0.2, 0.4, 0.6, 0.8]
iss = [1, 8]
eps = [10, 50, 100, 150]
tws = [5, 10, 15, 30, 50, 100]
lrs = [0.0001, 0.001, 0.005, 0.01, 0.05]

if __name__ == "__main__":
    i=0

    for ins in iss:
        for nl in nls:
            if nl > 1 :
                run(input_size = ins, num_layers = nl, dropout_rate = 0.2)
            else:
                run(input_size = ins, num_layers = nl)
        nl = pd.read_csv('results.csv').sort_values(by='mse').loc[0, 'layers']
        if nl == 1:
            dr = 0
        else:
            dr = 0.2

        for hs in hss:
            run(input_size = ins, num_layers = nl, hidden_size = hs, dropout_rate = dr)
        hs = pd.read_csv('results.csv').sort_values(by='mse').loc[0, 'hidden']

        for tw in tws:
            run(input_size = ins, num_layers = nl, hidden_size = hs, dropout_rate = dr, timeWindowToUse = tw)
        tw = pd.read_csv('results.csv').sort_values(by='mse').loc[0, 'Window']

        for lr in lrs:
            run(input_size = ins, num_layers = nl, hidden_size = hs, dropout_rate = dr, timeWindowToUse = tw, learning_rate = lr)
        lr = pd.read_csv('results.csv').sort_values(by='mse').loc[0, 'learning_rate']

        for ep in eps:
            run(input_size = ins, num_layers = nl, hidden_size = hs, dropout_rate = dr, timeWindowToUse = tw, learning_rate = lr, epoch = ep)
        ep = pd.read_csv('results.csv').sort_values(by='mse').loc[0, 'epochs']

        for bc in bcs:
            run(input_size = ins, num_layers = nl, hidden_size = hs, dropout_rate = dr, timeWindowToUse = tw, learning_rate = lr, epoch = ep, batch_size = bc)
        bc = pd.read_csv('results.csv').sort_values(by='mse').loc[0, 'batch_size']

        for dr in drs:
            if dr > 0 and ins == 1:
                continue
            run(input_size = ins, num_layers = nl, hidden_size = hs, dropout_rate = dr, timeWindowToUse = tw, learning_rate = lr, epoch = ep, batch_size = bc)
        dr = pd.read_csv('results.csv').sort_values(by='mse').loc[0, 'dropout']



    print("Work done !")