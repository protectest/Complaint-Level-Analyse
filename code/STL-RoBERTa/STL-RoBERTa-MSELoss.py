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
    batch_label = torch.zeros(len(batch_samples), 5).long()
    for id, sample in enumerate(batch_samples):
        batch_sentence.append(sample[1])
        batch_label[id][int(sample[0])] = 1
    x = tokenizer(
        batch_sentence,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    y = batch_label
    return x, y



#******************************************
#////build bidirectional RoBERTa module////
#******************************************
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
    # Forward propagation for loss
    predictions = model.forward(features.to(torch.int64).to(device))
    y_pred = [i for item in predictions for i in item]
    y_pred = [torch.max(i , 0)[1] for i in y_pred]
    y_true = [torch.max(i , 0)[1] for i in labels]
    # Calculate the correct number of predicted samples
    count = sum([1 for x, y in zip(y_pred, y_true) if x == y])
    # Computed loss function
    loss = loss_function(predictions.squeeze(), labels.float())


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
            lossi,correct_batch = train_step(model,batch_x.to(device), batch_y.to(device))
            list_loss.append(lossi)
            correct.append(correct_batch)
            progress_bar.update(1)
            progress_bar.set_description(f'loss: {lossi:>7f}')
        # Calculate the loss function for each round
        loss = np.mean(list_loss)
        # Accuracy of calculation
        accurency = sum(correct)/(dataloader.batch_size * len(correct))

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
            predictions = model.forward(batch_x.to(torch.int64).to(device))
            y_pred = [i for item in predictions for i in item]
            y_pred = [torch.max(i , 0)[1] for i in y_pred]
            y_true = [torch.max(i , 0)[1] for i in batch_y]
            if(epoch == epochs):
                y_show_true.append(y_true)
                y_show_pred.append(y_pred)

            count = sum([1 for x, y in zip(y_pred,y_true) if x == y])
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

step,(batch_x,batch_y) = next(enumerate(train_dataloader))

# Use MSELoss
loss_function = nn.MSELoss()

# Adam optimization function is used to define the learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_epochs = 50
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
train_accurency = go.Scatter(y=accurency, name="accurency")
trace_loss = go.Scatter(y=loss, name="loss")
fig1 = go.Figure(data=[train_accurency, trace_loss], layout=go.Layout(title='loss/accurency-epoch', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

test_accurency = go.Scatter(y=test_accurency, name='test accurency')
fig2 = go.Figure(data=[test_accurency], layout=go.Layout(title='test accurency', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

trace_true = go.Scatter(y=y_true, name='y_true')
fig3 = go.Figure(data=[trace_true], layout=go.Layout(title='y_true', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

ftrace_pred = go.Scatter(y=y_pred, name='y_pred')
fig4 = go.Figure(data=[ftrace_pred], layout=go.Layout(title='y_pred', xaxis=dict(title='X轴'), yaxis=dict(title='Y轴')))

fig = make_subplots(rows=2, cols=2, subplot_titles=('loss/accurency~epoch', 'test accurency~epoch', 'y_true', 'y_pred'))
fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig1.data[1], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)
fig.add_trace(fig3.data[0], row=2, col=1)
fig.add_trace(fig4.data[0], row=2, col=2)

fig.write_image("image.png",format='png')
