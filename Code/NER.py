import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import trange

# Load the dataset
data = pd.read_csv("NER_combined_dataset.csv", encoding='utf-8').fillna(method='ffill')

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
       # agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
       #                                                    s["POS"].values.tolist(),
        #                                                   s["Tag"].values.tolist())]
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence Id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["{}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)

sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
sentences[0]

#an example of sentence

labels = [[s[1] for s in sentence] for sentence in getter.sentences]
print(labels[0])
#BIO schema is followed in the datSET

tag_values = list(set(data["Tag"].values))
tag_values.append("PAD")
print(tag_values)

# Sort tag values
tag_values.sort()
tag2idx = {t: i for i, t in enumerate(tag_values)}
print(tag2idx)
#Padding is addded end of each sentence,

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,Dataset
from transformers import BertTokenizer, BertConfig

from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

torch.__version__

MAX_LEN = 75
bs = 16
#batch size = bs
# sentence length fixed to 75 i.e. 75 tokens
# but bert supports up to 512 tokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
#torch.cuda.get_device_name(0)

# Tokenize sentences and encode labels
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

# Pad and encode tokens and labels
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts], maxlen=MAX_LEN, dtype="long", value=0.0, truncating="post", padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels], maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post", dtype="long", truncating="post")
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

# Split data into train, validation, and test sets
train_inputs, test_inputs, train_tags, test_tags, train_masks, test_masks = train_test_split(input_ids, tags, attention_masks, random_state=2018, test_size=0.2)
train_inputs, val_inputs, train_tags, val_tags, train_masks, val_masks = train_test_split(train_inputs, train_tags, train_masks, random_state=2018, test_size=0.1)

# Convert data to PyTorch tensors
train_inputs, val_inputs, test_inputs = torch.tensor(train_inputs), torch.tensor(val_inputs), torch.tensor(test_inputs)
train_tags, val_tags, test_tags = torch.tensor(train_tags), torch.tensor(val_tags), torch.tensor(test_tags)
train_masks, val_masks, test_masks = torch.tensor(train_masks), torch.tensor(val_masks), torch.tensor(test_masks)

# Create data loaders
train_data = TensorDataset(train_inputs, train_masks, train_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

test_data = TensorDataset(test_inputs, test_masks, test_tags)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=bs)



from torchcrf import CRF

from transformers import BertModel,BertForTokenClassification
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

class BiLSTM_CRF_BERT(nn.Module):
    def __init__(self, bert_model, hidden_dim, tagset_size, dropout_prob=0.1):
        super(BiLSTM_CRF_BERT, self).__init__()
        self.bert_model = bert_model
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(bidirectional=True, num_layers=1, input_size=bert_model.config.hidden_size, hidden_size=hidden_dim // 2, batch_first=True, dropout=0.2)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, input_ids, attention_mask, tags):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        bert_hidden_states = outputs.hidden_states[-1]  # Get the last layer hidden states
        lstm_out, _ = self.lstm(bert_hidden_states)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        loss = -self.crf(tag_space, tags)
        predicted_tags = self.crf.decode(tag_space)  # Decode the most likely sequence of tags
        return loss, predicted_tags


# Instantiate BERT model
bert_model = BertForTokenClassification.from_pretrained("bert-base-cased")

# Instantiate the model
model = BiLSTM_CRF_BERT(bert_model, hidden_dim=256, tagset_size=len(tag2idx), dropout_prob=0.1)
model.to(device)

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    #no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=1e-4,
    eps=1e-8
)

#schduler to reduce learning rate linearly throughout the epochs
from transformers import get_linear_schedule_with_warmup

epochs = 20
max_grad_norm = 0.5

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

from tqdm import trange

train_losses = []
val_losses = []

# Training loop
for epoch in trange(epochs, desc="Epoch"):
    model.train()
    total_train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, mask, tags = batch
        optimizer.zero_grad()  # Clear gradients before computing the loss
        loss, _ = model(input_ids, mask, tags)
        total_train_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

    # Compute average training loss for the epoch
    average_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(average_train_loss)

    # Validation phase
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, mask, tags = batch
            loss, _ = model(input_ids, mask, tags)
            total_val_loss += loss.item()

    # Compute average validation loss for the epoch
    average_val_loss = total_val_loss / len(valid_dataloader)
    val_losses.append(average_val_loss)

    # Step the learning rate scheduler at the end of each epoch
    scheduler.step()

    # Print training and validation losses for the epoch
    print(f"Epoch {epoch + 1}:")
    print(f"  Training Loss: {average_train_loss:.4f}")
    print(f"  Validation Loss: {average_val_loss:.4f}")

# save the trained model
PATH = 'NER_save_dict.pth'
torch.save(model.state_dict(), PATH)
# save the trained model
PATH = 'NER_model.pth'
torch.save(model, PATH)

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model to evaluation mode
model.eval()

predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, mask, tags = batch
        _, pred_tags = model(input_ids, mask, tags)
        predictions.extend(pred_tags)
        true_labels.extend(tags.detach().cpu().numpy())

from sklearn.metrics import classification_report

# Convert tag indices to tag words
tag_idx2word = {idx: tag for tag, idx in tag2idx.items()}
pred_tags_words = [[tag_idx2word[idx] for idx in seq] for seq in predictions]
true_tags_words = [[tag_idx2word[idx] for idx in seq] for seq in true_labels]

# Remove 'PAD' tag
pred_tags_words_no_pad = [[tag for tag in seq if tag != 'PAD'] for seq in pred_tags_words]
true_tags_words_no_pad = [[tag for tag in seq if tag != 'PAD'] for seq in true_tags_words]

# Flatten the lists of lists
pred_tags_flat = [tag for seq in pred_tags_words_no_pad for tag in seq]
true_tags_flat = [tag for seq in true_tags_words_no_pad for tag in seq]

# Print classification report
print(classification_report(true_tags_flat, pred_tags_flat))

p = []
t = []
for i in pred_tags_flat:
  if i == 'O':
    p.append(i)
  else:
    p.append(i[2:])

for i in true_tags_flat:
  if i == 'O':
    t.append(i)
  else:
    t.append(i[2:])

from sklearn.metrics import classification_report
print(classification_report(t, p))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Assuming t and p are your true labels and predicted labels respectively

# Generate classification report
report = classification_report(t, p)

# Print classification report
print(report)

# Get class labels
labels = np.unique(t)

# Compute confusion matrix
cm = confusion_matrix(t, p, labels=labels)

# Plot confusion matrix
plt.figure(figsize=(15,13))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
image_name = f"confusion_matrix.png"
plt.savefig(image_name)
plt.show()

report = classification_report(t, p, digits=4, output_dict=True)
# Access the weighted F1 score, recall, and precision
f1_weighted = report['weighted avg']['f1-score']

recall_weighted = report['weighted avg']['recall']
precision_weighted = report['weighted avg']['precision']

# Print the results

print ('Weighted F1 Score: ', f1_weighted)
print ('Weighted Recall: ', recall_weighted)
print ('Weighted Precision: ', precision_weighted)

report = classification_report(t, p, digits=4, output_dict=True)
# Access the weighted F1 score, recall, and precision
f1_macro = report['macro avg']['f1-score']

recall_macro = report['macro avg']['recall']
precision_macro = report['macro avg']['precision']

# Print the results

print ('Macro F1 Score: ', f1_macro)
print ('Macro Recall: ', recall_macro)
print ('Macro Precision: ', precision_macro)
# Appending classification report, confusion matrix text, and confusion matrix image path to the same text file
with open("NER.txt", "a") as file:
    file.write("\n\nClassification Report:\n\n")
    file.write(report)
    file.write(f"\n\nWeighted Precision: {precision_weighted}")
    file.write(f"\n\nWeighted Recall: {recall_weighted}")
    file.write(f"\n\nWeighted F1 Score: {f1_weighted}")
    file.write(f"\n\Macro Precision: {precision_macro}")
    file.write(f"\n\Macro Recall: {recall_macro}")
    file.write(f"\n\Macro F1 Score: {f1_macro}")
