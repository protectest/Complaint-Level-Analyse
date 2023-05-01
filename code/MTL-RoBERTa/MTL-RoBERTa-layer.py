import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel,RobertaPreTrainedModel
from transformers import RobertaConfig
from tqdm.auto import tqdm
from plotly import graph_objects as go
from plotly.subplots import make_subplots



# load data
class Sequence_Dataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.data = self.load_data(annotations_file)


    def load_data(self, annotations_file):
        df = pd.read_csv(annotations_file,header=None,names=['id', 'text', 'lable', 'multilabel', 'domain'])
        Data = {}
        for idx, sentense in enumerate(np.array(list(zip(df['multilabel'],df['text'],df['lable'])))):
            Data[idx] = sentense
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


#*****************************
#////data process function////
#*****************************
#data tokenization
def collote_fn(batch_samples):
    batch_sentence = []
    batch_label_multi = torch.zeros(len(batch_samples), 5).long()
    batch_label_exp = torch.zeros(len(batch_samples), 5).long()
    for id, sample in enumerate(batch_samples):
        batch_sentence.append(sample[1])
        batch_label_multi[id][int(sample[0])] = 1
        batch_label_exp[id][int(sample[0])] = 1
    x = tokenizer(
        batch_sentence,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    y = batch_label_multi
    z = batch_label_exp
    return x, y, z



#******************************************
#////build bidirectional RoBERTa module////
#******************************************
class RobertaPreTrainedModel_tuning(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(768, 5)
        self.classifier2 = nn.Linear(768, 2)
        self.post_init()

    def forward(self, x):

        #shared model
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)

        # task1
        intensity = self.classifier1(cls_vectors)

        # task2
        exp = self.classifier2(cls_vectors)
        return intensity, exp


def train_step(model, features, labels1, labels2):
    # Forward propagation for loss
    predictions1, predictions2 = model.forward(features.to(torch.int64).to(device))
    y_pred1 = [i for item in predictions1 for i in item]
    y_pred1 = [torch.max(i , 0)[1] for i in y_pred1]
    y_true1 = [torch.max(i , 0)[1] for i in labels1]
    y_pred2 = [i for item in predictions2 for i in item]
    y_pred2 = [torch.max(i , 0)[1] for i in y_pred2]
    y_true2 = [torch.max(i , 0)[1] for i in labels2]

    # Calculate the correct number of predicted samples
    count1 = sum([1 for x, y in zip(y_pred1, y_true1) if x == y])
    count2 = sum([1 for x, y in zip(y_pred2, y_true2) if x == y])

    # Computed loss function
    loss1 = loss_function(predictions1.squeeze(), labels1.float())
    loss2 = loss_function(predictions2.squeeze(), labels2.float())

    # Apply diffurent weight of loss on diffurent task
    loss = 0.7*loss1 + 0.3*loss2

    # Inverse propagation to find the gradient
    loss.backward()
    # Parameter updating
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(),count1, count2


#********************
#////test a batch////
#********************
# features, labels = next(iter(train_dataloader))
# loss = train_step(model, features.float(), labels.float())
# print(loss)


# train model
def train_model(model, epochs, dataloader):
    model.train()
    progress_bar = tqdm(range(len(dataloader)*epochs))
    progress_bar.set_description(f'loss: {0:>7f}')

    loss_total = []
    accurency_total1 = []
    accurency_total2 = []

    for epoch  in range(1, epochs+1):
        list_loss = []
        correct1 = []
        correct2 = []
        for step,(batch_x,batch_y,batch_z) in enumerate(dataloader):
            lossi,correct_batch1,correct_batch2 = train_step(model,batch_x.to(device), batch_y.to(device), batch_z.to(device))
            list_loss.append(lossi)
            correct1.append(correct_batch1)
            correct2.append(correct_batch2)
            progress_bar.update(1)
            progress_bar.set_description(f'loss: {lossi:>7f}')
        # Calculate the loss function for each round
        loss = np.mean(list_loss)
        # Accuracy of calculation
        accurency1 = sum(correct1)/(dataloader.batch_size*len(correct1))
        accurency2 = sum(correct2)/(dataloader.batch_size*len(correct2))

        loss_total.append(loss)
        accurency_total1.append(accurency1)
        accurency_total2.append(accurency2)

        # Output the loss function every ten rounds
        if epoch % 10 == 0 and epoch != 0:
            print('epoch={} | loss={} '.format(epoch,loss))

    return loss_total, accurency_total1, accurency_total2



# test model
def test_model(model, dataloader,epochs):
    accurency_total1 = []
    accurency_total2 = []
    y_show_pred1 = []
    y_show_true1 = []
    y_show_pred2 = []
    y_show_true2 = []
    for epoch  in range(1, epochs+1):
        correct1 = []
        correct2 = []
        for step,(batch_x,batch_y,batch_z) in enumerate(dataloader):
            predictions1, predictions2 = model.forward(batch_x.to(torch.int64).to(device))
            y_pred1 = [i for item in predictions1 for i in item]
            y_pred1 = [torch.max(i , 0)[1] for i in y_pred1]
            y_true1 = [torch.max(i , 0)[1] for i in batch_y]
            y_pred2 = [i for item in predictions2 for i in item]
            y_pred2 = [torch.max(i , 0)[1] for i in y_pred2]
            y_true2 = [torch.max(i , 0)[1] for i in batch_z]

            if(epoch == epochs):
                y_show_true1.append(y_true1)
                y_show_pred1.append(y_pred1)
                y_show_true2.append(y_true2)
                y_show_pred2.append(y_pred2)

            count1 = sum([1 for x, y in zip(y_pred1,y_true1) if x == y])
            correct1.append(count1)
            count2 = sum([1 for x, y in zip(y_pred1,y_true1) if x == y])
            correct2.append(count2)
        accurency1 = sum(correct1) / (dataloader.batch_size * len(correct1))
        accurency_total1.append(accurency1)
        accurency2 = sum(correct2) / (dataloader.batch_size * len(correct2))
        accurency_total2.append(accurency2)

    return accurency_total1,accurency_total2,y_show_true1,y_show_pred1,y_show_true2,y_show_pred2



# Using GPU acceleration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Data set path
path = "..//complain_data.csv"

# model choose
checkpoint = "roberta-base"

# Use RoERTa tokenization
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)

# Load data
data = Sequence_Dataset(path)

# Split training and test data sets
train_dataset, test_dataset = torch.utils.data.random_split(data, [1239, 300])
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collote_fn, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collote_fn, drop_last=True)

# Initialization model
config = RobertaConfig.from_pretrained(checkpoint)
model = RobertaPreTrainedModel_tuning.from_pretrained(checkpoint, config=config).to(device)

# Use MSELoss
loss_function = nn.MSELoss()

# Adam optimization function is used to define the learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_epochs = 50
test_epochs = 10

# train model
loss,accurency1,accurency2 = train_model(model, train_epochs, train_dataloader)

# Prediction verification preview
test_accurency1,test_accurency2,y_true1,y_pred1,y_true2,y_pred2 = test_model(model, train_dataloader,test_epochs)
y_pred1 = sorted([i for item in y_pred1 for i in item])
y_pred1 = list(map(int,y_pred1))
y_true1 = sorted([i for item in y_true1 for i in item])
y_true1 = list(map(int,y_true1))
y_pred2 = sorted([i for item in y_pred2 for i in item])
y_pred2 = list(map(int,y_pred2))
y_true2 = sorted([i for item in y_true2 for i in item])
y_true2 = list(map(int,y_true2))

# visualization
train_accurency1 = go.Scatter(y=accurency1, name="accurency_intensity")
trace_loss1 = go.Scatter(y=loss, name="loss")
fig1 = go.Figure(data=[train_accurency1, trace_loss1], layout=go.Layout(title='loss/accurency-epoch', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

train_accurency2 = go.Scatter(y=accurency2, name="accurency_exp")
trace_loss2 = go.Scatter(y=loss, name="loss")
fig2 = go.Figure(data=[train_accurency2, trace_loss2], layout=go.Layout(title='loss/accurency-epoch', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

test_accurency1 = go.Scatter(y=test_accurency1, name='test accurency_intensity')
fig3 = go.Figure(data=[test_accurency1], layout=go.Layout(title='test accurency', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

test_accurency2 = go.Scatter(y=test_accurency2, name='test accurency_exp')
fig4 = go.Figure(data=[test_accurency2], layout=go.Layout(title='test accurency', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

trace_true1 = go.Scatter(y=y_true1, name='y_true')
fig5 = go.Figure(data=[trace_true1], layout=go.Layout(title='y_true', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

ftrace_pred1 = go.Scatter(y=y_pred1, name='y_pred')
fig6 = go.Figure(data=[ftrace_pred1], layout=go.Layout(title='y_pred', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

trace_true2 = go.Scatter(y=y_true2, name='y_true')
fig7 = go.Figure(data=[trace_true2], layout=go.Layout(title='y_true', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

ftrace_pred2 = go.Scatter(y=y_pred2, name='y_pred')
fig8 = go.Figure(data=[ftrace_pred2], layout=go.Layout(title='y_pred', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

fig = make_subplots(rows=2, cols=2, subplot_titles=('loss/accurency~epoch', 'test accurency~epoch', 'y_true', 'y_pred'))
fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig1.data[1], row=1, col=1)
fig.add_trace(fig3.data[0], row=1, col=2)
fig.add_trace(fig5.data[0], row=2, col=1)
fig.add_trace(fig6.data[0], row=2, col=2)

fig.write_image("image_intensity.png",format='png')

fig = make_subplots(rows=2, cols=2, subplot_titles=('loss/accurency~epoch', 'test accurency~epoch', 'y_true', 'y_pred'))
fig.add_trace(fig2.data[0], row=1, col=1)
fig.add_trace(fig2.data[1], row=1, col=1)
fig.add_trace(fig4.data[0], row=1, col=2)
fig.add_trace(fig7.data[0], row=2, col=1)
fig.add_trace(fig8.data[0], row=2, col=2)

fig.write_image("image_exp.png",format='png')


