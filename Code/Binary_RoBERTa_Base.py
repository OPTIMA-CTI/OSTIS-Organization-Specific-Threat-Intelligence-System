import torch
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc

from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig, AutoModel,RobertaTokenizer,RobertaModel,RobertaConfig
from sklearn.preprocessing import LabelEncoder

from time import sleep

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

pretrained_model = RobertaModel.from_pretrained('roberta-base')
config = RobertaConfig()

"""Set the Runtime to GPU and check and set cuda availability with the following snippet"""

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
device

"""Substitute dataset file name with your own"""
def experiment(seed):
	start_time = time.time()  # Record the start time
	df = pd.read_csv(r'Preprocessed_Final_OSTI_Binary.csv').dropna() #Change name here

	LABELS = len(df['Domain'].value_counts())

	#Encoding labels
	encoder = LabelEncoder()
	encoder.fit(df['Domain'])
	df['enc_Domain'] = encoder.transform(df['Domain'])

	df = df[['Domain', 'preprocess_description', 'enc_Domain']]
	LABELS

	"""# Model"""

	# Defining some key variables that will be used later on in the training
	MAX_LEN = 512
	TRAIN_BATCH_SIZE = 16
	VALID_BATCH_SIZE = 16
	EPOCHS = 20
	LEARNING_RATE = 1e-05

	class Triage(Dataset):
		def __init__(self, dataframe, tokenizer, max_len):
			self.len = len(dataframe)
			self.data = dataframe
			self.tokenizer = tokenizer
			self.max_len = max_len

		def __getitem__(self, index):
			sentence = str(self.data.preprocess_description[index])
			sentence = " ".join(sentence.split())
			inputs = self.tokenizer.encode_plus(
				sentence,
				None,
				add_special_tokens=True,
				max_length=self.max_len,
				padding='max_length',
				return_token_type_ids=True,
				truncation=True
			)
			ids = inputs['input_ids']
			mask = inputs['attention_mask']

			if 'enc_Domain' not in self.data:
				return {
				'ids': torch.tensor(ids, dtype=torch.long),
				'mask': torch.tensor(mask, dtype=torch.long)
				}

			return {
				'ids': torch.tensor(ids, dtype=torch.long),
				'mask': torch.tensor(mask, dtype=torch.long),
				'targets': torch.tensor(self.data.enc_Domain[index], dtype=torch.long)
			}

		def __len__(self):
			return self.len
	experiment_seed = seed
	#Split dataset into train and validation
	train_indices, test_indices = train_test_split(list(range(len(df.enc_Domain))), random_state = seed, test_size=0.2, stratify=df.enc_Domain)

	train_dataset = df.copy().drop(test_indices).reset_index(drop=True)
	test_dataset = df.copy().drop(train_indices).reset_index(drop=True)


	print("FULL Dataset: {}".format(df.shape))
	print("TRAIN Dataset: {}".format(train_dataset.shape))
	print("TEST Dataset: {}".format(test_dataset.shape))

	training_set = Triage(train_dataset, tokenizer, MAX_LEN)
	testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

	train_params = {'batch_size': TRAIN_BATCH_SIZE,
		        'shuffle': True,
		        'num_workers': 0
		        }

	test_params = {'batch_size': VALID_BATCH_SIZE,
		        'shuffle': True,
		        'num_workers': 0
		        }

	training_loader = DataLoader(training_set, **train_params)
	testing_loader = DataLoader(testing_set, **test_params)

	# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

	class RobertaClass(torch.nn.Module):
		def __init__(self, pretrained_model_name: str, num_classes: int = None, dropout: float = 0.5):
			super().__init__()
			config = RobertaConfig.from_pretrained(pretrained_model_name, output_hidden_states=True)
			self.model = RobertaModel.from_pretrained(pretrained_model_name, config=config).base_model #pick only the main body of the model
			#for param in self.model.parameters():
			#param.requires_grad = False
			self.pre_classifier = torch.nn.Linear(768, 768)
			self.dropout = torch.nn.Dropout(dropout)
			self.classifier = torch.nn.Linear(768, num_classes)

		def forward(self, input_ids, attention_mask):
			output_1 = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
			hidden_state = output_1[0]
			pooler = hidden_state[:, 0]
			pooler = self.pre_classifier(pooler)
			pooler = torch.nn.ReLU()(pooler)
			pooler = self.dropout(pooler)
			output = self.classifier(pooler)
			return output

	#LOAD
	model = RobertaClass('roberta-base', LABELS)	


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	# Creating the loss function and optimizer
	loss_function = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

	# Function to calcuate the accuracy of the model
	def calcuate_accu(big_idx, targets):
		n_correct = (big_idx==targets).sum().item()
		return n_correct

	torch.cuda.empty_cache()

	# Defining the training function on the 80% of the dataset for tuning the secbert model
	def train(epoch):
		tr_loss = 0
		n_correct = 0
		nb_tr_steps = 0
		examples = len(train_dataset)
		losses = [None] * len(training_loader)
		model.train()
		#loop = tqdm(enumerate(training_loader), total=len(training_loader), leave=False)
		for i, data in enumerate(training_loader, 0):
			#print(i)
			ids = data['ids'].to(device, dtype = torch.long)
			mask = data['mask'].to(device, dtype = torch.long)
			targets = data['targets'].to(device, dtype = torch.long)

			outputs = model(ids, mask)
			loss = loss_function(outputs, targets)

			losses[i] = loss.item()
			optimizer.zero_grad()
			loss.backward()
			# # When using GPU
			optimizer.step()

		print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")
		return


	for epoch in range(EPOCHS):
		train(epoch)

	def check_accuracy(loader, model):

		#pred = []
		num_correct = 0
		num_samples = 0
		model.eval()
		y_scores = []
		with torch.no_grad():
			for i, data in enumerate(loader, 0):
				x = data['ids'].to(device, dtype = torch.long)
				mask = data['mask'].to(device, dtype = torch.long)
				y = data['targets'].to(device, dtype = torch.long)

				scores = model(x, mask)
				_, predictions = scores.max(1)
				num_correct += (predictions == y).sum()
				num_samples += predictions.size(0)
				pred.append(predictions.cpu().numpy())
				y_test.append(y.cpu().numpy())
				y_scores.extend(F.softmax(scores, dim=1)[:, 1].cpu().numpy())

			print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

	pred=[]
	y_pred =[]
	y_test =[]
	y =[]

	"""Save the model for further tests"""

	check_accuracy(testing_loader, model)

	for i in range(len(pred)):
		t = pred[i]
		for j in range(len(t)):
			y_pred.append(t[j])

	for i in range(len(y_test)):
		t = y_test[i]
		for j in range(len(t)):
			y.append(t[j])

	from sklearn.metrics import classification_report
	print(classification_report(y, y_pred))

	#labels = list(encoder.inverse_transform([0,1,2,3,4,5]))

	from sklearn.metrics import classification_report,accuracy_score

	# Assuming you have true labels (y_true) and predicted labels (y_pred)
	# Calculate classification report
	report = classification_report(y, y_pred, digits=4, output_dict=True)

	# Access the weighted F1 score, recall, and precision
	f1_weighted = report['weighted avg']['f1-score']
	recall_weighted = report['weighted avg']['recall']
	precision_weighted = report['weighted avg']['precision']
	# Access the macro-averaged F1 score, recall, and precision
	f1_macro = report['macro avg']['f1-score']
	recall_macro = report['macro avg']['recall']
	precision_macro = report['macro avg']['precision']
	accuracy=accuracy_score(y,y_pred)

	# Print the results
	print("Weighted Precision:", precision_weighted)
	print("Weighted Recall:", recall_weighted)
	print("Weighted F1 Score:", f1_weighted)

	# Print the results
	print("Macro Precision:", f1_macro)
	print("Macro Recall:", recall_macro)
	print("Macro F1 Score:", precision_macro)
	print("Accuracy: ", accuracy)

	from sklearn.metrics import confusion_matrix
	import matplotlib.pyplot as plt
	import seaborn as sns
	# Update labels for confusion matrix
	cm_labels = ['Irrelevant', 'Relevant']
	fig, ax = plt.subplots(figsize=(7, 7))
	cm_array = confusion_matrix(y, y_pred)
	#cm_labels = np.unique(labels)
	cm_array_df = pd.DataFrame(cm_array,index=cm_labels, columns=cm_labels)
	sns.heatmap(cm_array_df, annot=True,
		    cbar=False, fmt='1d', cmap='Blues', ax=ax)
	ax.set_title('Confusion Matrix', loc='left', fontsize=16)
	ax.set_xlabel('Predicted',fontsize=16)
	ax.set_ylabel('Actual',fontsize=16)
	image_name = f"confusion_matrix_{experiment_seed}.png"
	plt.savefig(image_name)
    
	# Calculate and print execution time
	end_time = time.time()
	execution_time = end_time - start_time
	print(f"Execution Time: {execution_time} seconds")
	# Appending classification report, confusion matrix text, and confusion matrix image path to the same text file
	with open("RoBERTa_Base_result.txt", "a") as file:
		file.write("\n\nRoBERTa_Base Result:\n\n")
		file.write(f"\n\nSeed Value: {experiment_seed}")
		file.write("\n\nClassification Report:\n\n")
		file.write(classification_report(y, y_pred))
		file.write(f"\n\nWeighted Precision: {precision_weighted}")
		file.write(f"\n\nWeighted Recall: {recall_weighted}")
		file.write(f"\n\nWeighted F1 Score: {f1_weighted}")
		file.write(f"\n\Macro Precision: {precision_macro}")
		file.write(f"\n\Macro Recall: {recall_macro}")
		file.write(f"\n\Macro F1 Score: {f1_macro}")
		file.write(f"\n\nAccuracy: {accuracy}")
		file.write("\n\nConfusion Matrix Image Path:\n\n")
		file.write(image_name)
		file.write(f"\n\nExecution Time: {execution_time} seconds")
   

experiment(3832)
experiment(479)
experiment(3153)
experiment(6169)
experiment(8891)
experiment(1761)
experiment(6025)
experiment(7871)
experiment(1659)
experiment(7604)


