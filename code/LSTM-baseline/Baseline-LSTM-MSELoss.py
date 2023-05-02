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
        for idx, sentense in enumerate(np.array(list(zip(df['multilabel'],df['text'])))):
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
    batch_label = torch.zeros(16, 5).long()  # Batch size
    for id, sample in enumerate(batch_samples):
        batch_sentence.append(sample[1])
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



#***************************************
#////build bidirectional LSTM module////
#***************************************
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
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)

        output, _ = self.lstm(input_seq.float().to(device),(h_0,c_0))
        pred= self.fc(output)
        pred= pred[:, -1, :]

        pred = torch.stack([pred], dim=0)

        return pred



def train_step(model, features, labels):
    # Forward propagation for loss
    predictions = model.forward(features.to(torch.int64))
    y_pred = [i for item in predictions for i in item]
    y_pred = [torch.max(i , 0)[1] for i in y_pred]
    y_true = [torch.max(i , 0)[1] for i in labels]
    # Calculate the correct number of predicted samples
    count = sum([1 for x, y in zip(y_pred, y_true) if x == y])
    # Computed loss function
    loss = loss_function(predictions.squeeze(), labels.float().to(device))

    # Inverse propagation to find the gradient
    loss.backward()
    # Parameter updating
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(),count

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
        # Calculate the loss function for each round
        loss = np.mean(list_loss)
        # Accuracy of calculation
        accurency = sum(correct)/(dataloader.batch_size*len(correct))

        loss_total.append(loss)
        accurency_total.append(accurency)

        # Output the loss function every ten rounds
        if epoch % 10 == 0 and epoch != 0:
            print('epoch={} | loss={} '.format(epoch,loss))

    return loss_total, accurency_total



# test model
def test_model(model, dataloader,epochs):
    accurency_total = []
    y_show_pred = []
    y_show_true = []
    for epoch  in range(1, epochs+1):
        correct = []
        for step,(batch_x,batch_y) in enumerate(dataloader):
            predictions = model.forward(batch_x.float())
            y_pred = [i for item in predictions for i in item]
            y_pred = [torch.max(i , 0)[1] for i in y_pred]
            y_true = [torch.max(i, 0)[1] for i in batch_y]
            if(epoch == epochs):
                y_show_true.append(y_true)
                y_show_pred.append(y_pred)
            count = sum([1 for x, y in zip(y_pred, y_true) if x == y])
            correct.append(count)
        accurency = sum(correct) / (dataloader.batch_size * len(correct))
        accurency_total.append(accurency)

    return accurency_total,y_show_true,y_show_pred



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
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collote_fn, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collote_fn, drop_last=True)

# Initialization model
model = LSTM(512, 1024, 5, 5, 4, bidirectional=False).to(device)

# Use the cross entropy loss function
loss_function = nn.MSELoss()

# Adam optimization function is used to define the learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_epochs = 100
test_epochs = 10

# train model
loss,accurency = train_model(model, train_epochs, train_dataloader)

# Prediction verification preview
test_accurency,y_true,y_pred = test_model(model, train_dataloader,test_epochs)
y_pred = sorted([i for item in y_pred for i in item])
y_pred = list(map(int,y_pred))
y_true = sorted([i for item in y_true for i in item])
y_true = list(map(int,y_true))

# visualization
train_accurency = go.Scatter(y=accurency, name="accuracy")
trace_loss = go.Scatter(y=loss, name="loss")
fig1 = go.Figure(data=[train_accurency, trace_loss], layout=go.Layout(title='loss/accuracy-epoch', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

test_accurency = go.Scatter(y=test_accurency, name='test accuracy')
fig2 = go.Figure(data=[test_accurency], layout=go.Layout(title='test accuracy', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

trace_true = go.Scatter(y=y_true, name='y_true')
fig3 = go.Figure(data=[trace_true], layout=go.Layout(title='y_true', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

ftrace_pred = go.Scatter(y=y_pred, name='y_pred')
fig4 = go.Figure(data=[ftrace_pred], layout=go.Layout(title='y_pred', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

fig = make_subplots(rows=2, cols=2, subplot_titles=('loss/accuracy~epoch', 'test accuracy~epoch', 'y_true', 'y_pred'))
fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig1.data[1], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)
fig.add_trace(fig3.data[0], row=2, col=1)
fig.add_trace(fig4.data[0], row=2, col=2)

fig.write_image("image.png",format='png')


