import pandas as pd
data = pd.read_csv('../../dataset.csv')

data['label'] = data['MAIN_CAT'].apply(lambda x: 0 if x == 'ANOIMPACT' else 1)
data['len'] = data['MAIN_CAT'].apply(lambda x:len(x.split(',')))
data = data[data['len']==1]

data = data[data['MAIN_CAT']!='ANOIMPACT']
data = data.sample(frac=1, random_state = 1)
data.index = range(len(data))


# resample and balance data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['MAIN_CAT'])
ddd = data
#dd = data[data['MAIN_CAT'] != 'ANOIMPACT']
#ddd = pd.concat([data[data['MAIN_CAT'] == 'ANOIMPACT'].sample(n=len(dd), random_state = 1),dd])
test = data[data['TRAIN_TEST'] == 'TEST']
train = ddd[ddd['TRAIN_TEST'] == 'TRAIN']
#test = ddd[ddd['TRAIN_TEST'] == 'TEST']
val = train.sample(frac = 0.15, random_state = 10)
tra = train[~train.index.isin(val.index)]
dd = tra[tra['MAIN_CAT'] != 'ANOIMPACT']
max_size = dd['MAIN_CAT'].value_counts().max()
max_size = dd['MAIN_CAT'].value_counts().max()
max_size
lst = [dd]
for class_index, group in dd.groupby('MAIN_CAT'):
    lst.append(group.sample(max_size-len(group), replace=True))
frame_new = pd.concat(lst)
tra = frame_new
tra.index = range(len(tra))
tra = tra.sample(frac = 1, ignore_index = True, random_state = 1)


train_texts = tra.TEXT.tolist()
train_labels = tra.label.tolist()
val_texts = val.TEXT.tolist()
val_labels = val.label.tolist()


# fine-tuning
from transformers import BertTokenizer, BertForSequenceClassification 
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd
import numpy as np
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
import random
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased') # change the tokenizer and "roberta-base"
model = BertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=8).to("cuda") 


max_length = 200
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

class torch_ds(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = torch_ds(train_encodings, train_labels)
valid_dataset = torch_ds(valid_encodings, val_labels)


from sklearn.metrics import f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
    acc = f1_score(labels, preds, average = 'weighted')
    return {
      'acc': acc,
    }


from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir='./mainmodel_impact',          # output directory
    num_train_epochs=6,
    save_total_limit=3,
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_ratio=0.05,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs_impact_main',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    logging_steps=200,               # log & save weights each logging_steps
    save_steps=200,
    learning_rate = 1e-5,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)


trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

trainer.train()