import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from tqdm.auto import tqdm
from plotly import graph_objects as go
from plotly.subplots import make_subplots



tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class Sequence_Dataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.data = self.load_data(annotations_file)


    def load_data(self, annotations_file):
        df = pd.read_csv(annotations_file,header=None,names=['id', 'text', 'lable', 'multilabel', 'domain'])
        Data = {}
        for idx, sentense in enumerate(np.array(list(zip(df['multilabel'],df['text'])))):
            Data[idx] = sentense
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collote_fn(batch_samples):
    batch_sentence = []
    batch_label = torch.zeros(4, 5).long()
    for id, sample in enumerate(batch_samples):
        batch_sentence.append(sample[1])
        # batch_label.append(int(sample[0]))
        batch_label[id][int(sample[0])] = 1
    x = [tokenizer.encode(
        sentence,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ) for sentence in batch_sentence]
    x = torch.tensor([t.numpy() for t in x])
    y = batch_label
    return x, y

train_data = Sequence_Dataset("..\complaint_severity_data.csv")
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn, drop_last=True)
# print(next(enumerate(train_dataloader)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm = nn.LSTM(hidden_size=self.hidden_size, num_layers=self.num_layers, input_size=input_size, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(self.hidden_size*self.num_directions, self.output_size).to(device)

    def forward(self, input_seq):
        # print(input_seq.shape)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)

        # print(batch_size,seq_len)
        # print(h_0,c_0)
        # print(h_0.shape,c_0.shape)
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)

        # output, _ = self.lstm(input_seq, (h_0, c_0))

        # print(packed_embedded)
        # for i in packed_embedded.data:
        #     print(i.shape)
        # exit()


        output, _ = self.lstm(input_seq.float(),(h_0,c_0))
        preds = []
        pred= self.fc(output)
        pred= pred[:, -1, :]

        # pred = torch.cat([pred1, pred2], dim=0)
        pred = torch.stack([pred], dim=0)
        # print(pred.shape)

        return pred


# for step,(batch_x,batch_y) in enumerate(train_dataloader):
#     model = LSTM(512, 64, 5, 1, 4,1).to(device)
#     model(batch_x.float())


def train_step(model, features, labels):
    # 正向传播求损失
    predictions = model.forward(features.to(torch.int64))
    # predictions = model.forward(features.float())
    # print(labels)
    # print(labels.shape)
    y_pred = [i for item in predictions for i in item]
    y_pred = [np.argmax(i.detach().numpy()) for i in y_pred]
    # print(labels)
    # y_true = [i for item in labels for i in item]
    # print(y_true)
    y_true = [np.argmax(i.detach().numpy()) for i in labels]
    # print(y_pred)
    # print(y_true)
    # exit()

    count = sum([1 for x, y in zip(y_pred, y_true) if x == y])

    # labels = labels.reshape(1,4,1)
    labels = labels.reshape(1,4,5)
    loss = loss_function(predictions, labels.float())
    # loss = loss_function(predictions, labels.to(torch.int64))
    # 反向传播求梯度
    loss.backward()
    # 参数更新
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(),count


# 测试一个batch
# features, labels = next(iter(train_dataloader))
# loss = train_step(model, features.float(), labels.float())
# print(loss)

#%%
# 训练模型
def train_model(model, epochs, dataloader):
    model.train()
    progress_bar = tqdm(range(len(dataloader)*epochs))
    progress_bar.set_description(f'loss: {0:>7f}')

    loss_total = []
    accurency_total = []

    for epoch  in range(1, epochs+1):
        list_loss = []
        correct = []
        for step,(batch_x,batch_y) in enumerate(dataloader):
            lossi,correct_batch = train_step(model,batch_x, batch_y)
            list_loss.append(lossi)
            correct.append(correct_batch)
            progress_bar.update(1)
            progress_bar.set_description(f'loss: {lossi:>7f}')
        loss = np.mean(list_loss)
        accurency = sum(correct)/(4*len(correct))

        loss_total.append(loss)
        accurency_total.append(accurency)

        if epoch % 10 == 0 and epoch != 0:
            print('epoch={} | loss={} '.format(epoch,loss))

    return loss_total, accurency_total


# 测试模型
def test_model(model, dataloader):
    y_pred = []
    for step,(batch_x,batch_y) in enumerate(dataloader):
        y_pred.append(model.forward(batch_x.float()))
    return y_pred



model = LSTM(512, 64, 5, 5, 4, bidirectional=False).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss,accurency = train_model(model, 500, train_dataloader)


#%%
# 预测验证预览
y_pred = test_model(model, train_dataloader)
y_pred = [i for items in y_pred for item in items for i in item]
# print( [i.detach().numpy() for i in y_pred])
# print(y_pred[0])
# print(len(y_pred))
y_pred = [np.argmax(i.detach().numpy()) for i in y_pred]
# print("//////////////////////////////////////////////////////////")
# print(y_pred[0])
# print(len(y_pred))

df = pd.read_csv("..\complaint_severity_data.csv", header=None, names=['id', 'text', 'lable', 'multilabel', 'domain'])
y_true = df['multilabel']


trace_accurency = go.Scatter(y=accurency, name="accurency")
trace_loss = go.Scatter(y=loss, name="loss")
fig1 = go.Figure(data=[trace_accurency, trace_loss], layout=go.Layout(title='第一张图', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

trace_true = go.Scatter(y=y_true, name='y_true')
ftrace_pred = go.Scatter(y=y_pred, name='y_pred')
fig2 = go.Figure(data=[trace_true, ftrace_pred], layout=go.Layout(title='第一张图', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))


fig = make_subplots(rows=1, cols=2, subplot_titles=('第一张图', '第二张图'))
fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig1.data[1], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)
fig.add_trace(fig2.data[1], row=1, col=2)

fig.show()
#
#
# # def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
# #     progress_bar = tqdm(range(len(dataloader)))
# #     progress_bar.set_description(f'loss: {0:>7f}')
# #     finish_step_num = (epoch - 1) * len(dataloader)
# #
# #     model.train()
# #     for step, (X, y) in enumerate(dataloader, start=1):
# #         X, y = X.to(device), y.to(device)
# #         pred = model(X)
# #         loss = loss_fn(pred, y)
# #
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
# #         lr_scheduler.step()
# #
# #         total_loss += loss.item()
# #         progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
# #         progress_bar.update(1)
# #     return total_loss
# #
# #
# # def test_loop(dataloader, model, mode='Test'):
# #     assert mode in ['Valid', 'Test']
# #     size = len(dataloader.dataset)
# #     correct = 0
# #
# #     model.eval()
# #     with torch.no_grad():
# #         for X, y in dataloader:
# #             X, y = X.to(device), y.to(device)
# #             pred = model(X)
# #             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
# #
# #     correct /= size
# #     print(f"{mode} Accuracy: {(100 * correct):>0.1f}%\n")
