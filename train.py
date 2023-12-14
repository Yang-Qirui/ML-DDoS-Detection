from lstm_cloud import LSTM_CLOUD
import argparse
import torch
import torch.nn as nn
from utils import DDoSDataset, load_npy
from torch.utils.data import DataLoader


def train(args):
    model_dict = {
        "lstm_cloud": LSTM_CLOUD(feature_num=args.feature_num, hidden_size=args.hidden_size, seq_len=args.seq_len, hidden_num=args.hidden_num),
    }
    model = model_dict[args.model_name]
    loss_fn_dict = {
        "mse": nn.MSELoss()
    }
    # model.load_state_dict(torch.load("./model.pt"))
    loss_fn = loss_fn_dict[args.loss_fn]
    data = load_npy(args.train_dir)
    # test_data = load_npy(args.test_dir)
    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    # train_data = data[:int(0.1 * len(data))]
    # test_data = data[int(0.1 * len(data)):]

    train_dataset = DDoSDataset(train_data)
    test_dataset = DDoSDataset(test_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for x, y in train_dataloader:
            # has_nan_x = torch.isnan(x).any()
            # has_nan_y = torch.isnan(y).any()
            # # print(x,y)
            # if has_nan_x or has_nan_y:
            #     assert 0

            prediction = model(x.float())
            # print(prediction)
            loss = loss_fn(prediction, y.float())
            total_loss += loss.item()
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.epoch}, Loss: {total_loss/len(train_dataloader)}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_dataloader:
                features, labels = batch
                
                outputs = model(features.float())

                # Assuming a multi-class classification problem
                pred_out = torch.where(outputs > 0.5, torch.tensor(1), torch.tensor(0))

                total += labels.shape[0] * labels.shape[1]
                correct += (pred_out == labels).sum().item()

        accuracy = 100 * correct / total
        
        print(f"Test Accuracy: {accuracy} %")
    # torch.save(model.state_dict(), "./model.pt")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-model_name", help="models name. [lstm_cloud]", default="lstm_cloud", type=str)
    arg_parser.add_argument("-feature_num", help="# of features.", default=2, type=int)
    arg_parser.add_argument("-hidden_size", help="dim of lstm hidden layer output.", default=32, type=int)
    arg_parser.add_argument("-hidden_num", help="# of lstm hidden layers.", default=2, type=int)
    arg_parser.add_argument("-loss_fn", help="loss functions. [mse]", default="mse", type=str)
    arg_parser.add_argument("-lr", help="learning rate", default=0.001, type=float)
    arg_parser.add_argument("-train_dir", help="the path of training directory", default="./dataset/CICDDoS2019/01-12", type=str)
    arg_parser.add_argument("-test_dir", help="the path of testing directory", default="./dataset/CICDDoS2019/03-11", type=str)
    arg_parser.add_argument("-seq_len", help="sequence length of LSTM", default=5, type=int)
    arg_parser.add_argument("-epoch", help="epoch number", default=20, type=int)
    arg_parser.add_argument("-batch_size", help="batch size", default=32, type=int)



    args = arg_parser.parse_args()

    train(args)