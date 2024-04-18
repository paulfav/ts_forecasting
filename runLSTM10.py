import argparse
from LSTMmodel10 import *



part_train =  len(pd.read_csv('train_normalised.csv')) / (len(pd.read_csv('train_normalised.csv')) + len(pd.read_csv('test_normalised.csv')))
part_valid = 0.2 * part_train
part_train =  0.8 * part_train

def getfeatures(input_size = 1, timeWindowToUse = 15, output_size = 50):
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
        for i in range(0, output_size):
            df["y_{}".format(i)] = df["y"].shift(-i)
        df.dropna(inplace=True)
        y = df[["y_{}".format(i) for i in range(output_size)]].to_numpy()
        x = df[["x_{}".format(i) for i in range(timeWindowToUse, 0, -1)]].to_numpy()
        print(y)
        return x, y
    
    else :
        df = pd.concat([data_train[data_train.columns[4::6]], data_test[data_test.columns[4::6]]])
        x = np.array([df.shift(i).to_numpy() for i in range(timeWindowToUse, 0, -1)])

        x = np.concatenate(x, axis=1).reshape(-1, timeWindowToUse, 8)
        y = df["close_" + stock_name].to_numpy().reshape(-1,1)
        return x[timeWindowToUse:], y[timeWindowToUse:]

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--multiple_input', type=bool, default=False, help='multiple input')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--timeWindowToUse', type=int, default=15, help='time window to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--output_size', type=float, default=10, help='output size')

    args = parser.parse_args()



    if not args.multiple_input:
        x, y  = getfeatures(1, args.timeWindowToUse)
        input_size = 1
        print(y)
    else :
        x, y  = getfeatures(8, args.timeWindowToUse)
        input_size = 8
        print("????")

    train_model(x, y, batch_size = args.batch_size, hidden_size = args.hidden_size, num_layers = args.num_layers,
                 dropout_rate = args.dropout_rate, input_size = input_size, part_train = part_train, part_valid = part_valid,
                  learning_rate= args.learning_rate, epoch = args.epoch, output_size = int(args.output_size))
    true_Y, pred_Y = test_model(x, y, hidden_size = args.hidden_size, num_layers = args.num_layers, dropout_rate = args.dropout_rate, input_size = input_size, part_train = part_train, part_valid = part_valid, output_size=int(args.output_size))
    mse = mean_squared_error(true_Y, pred_Y)


    res = pd.read_csv('results.csv')
    dicToAdd = {'Model': 'LSTM','input_size': input_size,'output_size': args.output_size,'Window': args.timeWindowToUse,'layers': args.num_layers,'hidden': args.hidden_size,
                    'batch_size' : args.batch_size, 'dropout': args.dropout_rate, 'learning_rate': args.learning_rate,
                    'epochs': args.epoch, 'mse': mse}
    res = pd.concat([res, pd.DataFrame.from_records([dicToAdd])])
    print(dicToAdd)
    print("MSE: ", mse) 
    res.to_csv('results.csv', index=False)