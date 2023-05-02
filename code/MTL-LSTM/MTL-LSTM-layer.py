import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
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
    batch_label_multi = []
    batch_label_exp = []
    for id, sample in enumerate(batch_samples):
        batch_sentence.append(sample[1])
        batch_label_multi.append(int(float(sample[0])))
        batch_label_exp.append(int(float(sample[2])))
    x = [tokenizer.encode(
        sentence,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ) for sentence in batch_sentence]
    x = torch.tensor([t.numpy() for t in x])
    y = torch.tensor(batch_label_multi)
    z = torch.tensor(batch_label_exp)
    return x, y, z



#***************************************
#////build bidirectional LSTM module////
#***************************************
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size1, output_size2, batch_size, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_size = batch_size
        self.output_size1 = output_size1
        self.output_size2 = output_size2
        self.lstm = nn.LSTM(hidden_size=self.hidden_size, num_layers=self.num_layers, input_size=input_size, batch_first=True, bidirectional=bidirectional)
        self.task1 = nn.Linear(self.hidden_size*self.num_directions, self.output_size1).to(device)
        self.task2 = nn.Linear(self.hidden_size*self.num_directions, self.output_size2).to(device)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)

        # shared layer
        output, _ = self.lstm(input_seq.float().to(device),(h_0,c_0))

        #task1
        pred1= self.task1(output)
        pred1= pred1[:, -1, :]
        pred1 = torch.stack([pred1], dim=0)

        #task2
        pred2= self.task2(output)
        pred2= pred2[:, -1, :]
        pred2 = torch.stack([pred2], dim=0)

        return pred1, pred2



def train_step(model, features, labels1, labels2):
    # Forward propagation for loss
    predictions1, predictions2 = model.forward(features.to(torch.int64))
    y_pred1 = [i for item in predictions1 for i in item]
    y_pred1 = [torch.max(i , 0)[1] for i in y_pred1]
    y_true1 = labels1
    # Calculate the correct number of predicted samples
    count1 = sum([1 for x, y in zip(y_pred1, y_true1) if x == y])

    y_pred2 = [i for item in predictions2 for i in item]
    y_pred2 = [torch.max(i , 0)[1] for i in y_pred2]
    y_true2 = labels2

    # Calculate the correct number of predicted samples
    count2 = sum([1 for x, y in zip(y_pred2, y_true2) if x == y])

    # Computed loss function
    loss1 = loss_function1(predictions1.squeeze(), labels1.to(torch.int64).to(device))
    loss2 = loss_function2(predictions2.squeeze(), labels2.to(torch.int64).to(device))

    # Apply diffurent weight of loss on diffurent task
    loss = 0.7*loss1 + 0.3*loss2

    # Inverse propagation to find the gradient
    loss.backward()
    # Parameter updating
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(), count1, count2

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
            lossi,correct_batch1,correct_batch2 = train_step(model, batch_x, batch_y, batch_z)
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
            predictions1, predictions2 = model.forward(batch_x.float())
            y_pred1 = [i for item in predictions1 for i in item]
            y_pred1 = [torch.max(i , 0)[1] for i in y_pred1]
            y_true1 = batch_y
            y_pred2 = [i for item in predictions2 for i in item]
            y_pred2 = [torch.max(i , 0)[1] for i in y_pred2]
            y_true2 = batch_z

            if(epoch == epochs):
                y_show_true1.append(y_true1)
                y_show_pred1.append(y_pred1)
                y_show_true2.append(y_true2)
                y_show_pred2.append(y_pred2)

            count1 = sum([1 for x, y in zip(y_pred1, y_true1) if x == y])
            correct1.append(count1)
            count2 = sum([1 for x, y in zip(y_pred2, y_true2) if x == y])
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

# Use RoERTa tokenization
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load data
data = Sequence_Dataset(path)

# Split training and test data sets
train_dataset, test_dataset = torch.utils.data.random_split(data, [1239, 300])
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collote_fn, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collote_fn, drop_last=True)

# Initialization model
model = LSTM(input_size=512, hidden_size=64, num_layers=5, output_size1=5, output_size2=2, batch_size=4, bidirectional=False).to(device)

# Use the cross entropy loss function
loss_function1 = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1]).to(device))
loss_function2 = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1]).to(device))

# Adam optimization function is used to define the learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_epochs = 5
test_epochs = 1

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
train_accurency1 = go.Scatter(y=accurency1, name="accuracy_intensity")
trace_loss1 = go.Scatter(y=loss, name="loss")
fig1 = go.Figure(data=[train_accurency1, trace_loss1], layout=go.Layout(title='loss/accuracy-epoch', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

train_accurency2 = go.Scatter(y=accurency2, name="accuracy_exp")
trace_loss2 = go.Scatter(y=loss, name="loss")
fig2 = go.Figure(data=[train_accurency2, trace_loss2], layout=go.Layout(title='loss/accuracy-epoch', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

test_accurency1 = go.Scatter(y=test_accurency1, name='test accuracy_intensity')
fig3 = go.Figure(data=[test_accurency1], layout=go.Layout(title='test accuracy', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

test_accurency2 = go.Scatter(y=test_accurency2, name='test accuracy_exp')
fig4 = go.Figure(data=[test_accurency2], layout=go.Layout(title='test accuracy', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

trace_true1 = go.Scatter(y=y_true1, name='y_true')
fig5 = go.Figure(data=[trace_true1], layout=go.Layout(title='y_true', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

ftrace_pred1 = go.Scatter(y=y_pred1, name='y_pred')
fig6 = go.Figure(data=[ftrace_pred1], layout=go.Layout(title='y_pred', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

trace_true2 = go.Scatter(y=y_true2, name='y_true')
fig7 = go.Figure(data=[trace_true2], layout=go.Layout(title='y_true', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

ftrace_pred2 = go.Scatter(y=y_pred2, name='y_pred')
fig8 = go.Figure(data=[ftrace_pred2], layout=go.Layout(title='y_pred', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

fig = make_subplots(rows=2, cols=2, subplot_titles=('loss/accuracy~epoch', 'test accuracy~epoch', 'y_true', 'y_pred'))
fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig1.data[1], row=1, col=1)
fig.add_trace(fig3.data[0], row=1, col=2)
fig.add_trace(fig5.data[0], row=2, col=1)
fig.add_trace(fig6.data[0], row=2, col=2)

fig.write_image("image_intensity.png",format='png')

fig = make_subplots(rows=2, cols=2, subplot_titles=('loss/accuracy~epoch', 'test accuracy~epoch', 'y_true', 'y_pred'))
fig.add_trace(fig2.data[0], row=1, col=1)
fig.add_trace(fig2.data[1], row=1, col=1)
fig.add_trace(fig4.data[0], row=1, col=2)
fig.add_trace(fig7.data[0], row=2, col=1)
fig.add_trace(fig8.data[0], row=2, col=2)

fig.write_image("image_exp.png",format='png')


