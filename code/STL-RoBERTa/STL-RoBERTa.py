import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel,RobertaPreTrainedModel
from transformers import RobertaConfig
from tqdm.auto import tqdm
from plotly import graph_objects as go
from plotly.subplots import make_subplots



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
    batch_label = torch.zeros(len(batch_samples), 5).long()
    # batch_label = []
    for id, sample in enumerate(batch_samples):
        batch_sentence.append(sample[1])
        # batch_label.append(int(sample[0]))
        batch_label[id][int(sample[0])] = 1
    x = tokenizer(
        batch_sentence,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    y = batch_label
    return x, y



class RobertaPreTrainedModel_tuning(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 5)
        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits


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
    # labels = labels.reshape(1,4,5)
    # labels = labels.reshape(1,len(features),5)
    # labels = labels.reshape(1,5).squeeze()
    # loss = loss_function(predictions, labels.float())
    # print(predictions)
    # print(labels)

    loss = loss_function(predictions.squeeze(), labels.float())


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

path = "..\complain_data.csv"
checkpoint = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
train_data = Sequence_Dataset(path)

dataloader = DataLoader(train_data, batch_size=32)
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn, drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

config = RobertaConfig.from_pretrained(checkpoint)
model = RobertaPreTrainedModel_tuning.from_pretrained(checkpoint, config=config).to(device)

step,(batch_x,batch_y) = next(enumerate(train_dataloader))

# # loss_function = nn.CrossEntropyLoss(weight=torch.Tensor([1, 5, 6, 10, 10]))
# # [0.05, 0.25, 0.3, 0.45, 0.45]
#
# loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1]).to(device))
#
loss_function = nn.MSELoss()
# #4:200
# #3:220
# #2:380
# #1:430
# #0:2200
#
# # a = torch.tensor([[-0.1805, -0.0893, 0.1623, 0.0313, 0.0804]])
# # b = torch.tensor([1])
# # loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.05, 0.25, 0.3, 0.45, 0.45]).to(device))
# # print(loss_function(a, b))
# # exit()
#
# # from torch import autograd
# # loss = nn.CrossEntropyLoss()
# # input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
# # print(input)
# # print(input.shape)
# # target = autograd.Variable(torch.LongTensor(3).random_(5))
# # print(target)
# # output = loss(input, target)
# # output.backward()
# # exit()
#
#
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss,accurency = train_model(model, 50, train_dataloader)
#
#
# #%%
# # 预测验证预览
y_pred = test_model(model, train_dataloader)
y_pred = [i for items in y_pred for item in items for i in item]
# print( [i.detach().numpy() for i in y_pred])
# print(y_pred[0])
# print(len(y_pred))
y_pred = [np.argmax(i.detach().numpy()) for i in y_pred]
# print("//////////////////////////////////////////////////////////")
# print(y_pred[0])
# print(len(y_pred))

df = pd.read_csv(path, header=None, names=['id', 'text', 'lable', 'multilabel', 'domain'])
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
