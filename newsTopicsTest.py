import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from numpy import array
from numpy import asarray
from numpy import zeros
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import sys
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.utils import plot_model
from sklearn.svm import LinearSVC
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import KFold
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import *
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
seed = 7
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'data_files')
MAX_SEQUENCE_LENGTH = 600
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

df = pd.read_csv("mydata.csv",encoding = "utf-8")
df['category_id'] = df['category'].factorize()[0]
labels=df['category_id']
category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

contenta=df['content'].values
print('Processing text dataset')

embeddings_index = {}
# Word embeddings can be found at https://nlp.stanford.edu/projects/glove/

with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs



texts = []  
labels_index = {} 
labels2 = [] 
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fname2=fname[:-4]
            if fname2.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n') 
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels2.append(label_id)


texts=contenta
labels2=labels
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

data2 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels2 = to_categorical(np.asarray(labels2))


indices = np.arange(data2.shape[0])

np.random.shuffle(indices)
train_indices=indices[:1513]
test_indices=indices[1513:]

train_data_ff2 = data2[train_indices]
train_labels_ff2 = labels2[train_indices]
test_data_ff2 = data2[test_indices]
test_labels_ff2 = labels2[test_indices]



num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)



# train a 1D convnet with global maxpooling
def convModel():
  sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  embedded_sequences = embedding_layer(sequence_input)
  x = Conv1D(32, 5, activation='relu')(embedded_sequences)
  x = MaxPooling1D(5)(x)
  x = Conv1D(32, 5, activation='relu')(x)
  x = MaxPooling1D(5)(x)
  x = Conv1D(32, 5, activation='relu')(x)
  x = GlobalMaxPooling1D()(x)
  x = Dense(32, activation='relu')(x)
  preds = Dense(len(labels_index), activation='softmax')(x)
  model = Model(sequence_input, preds)
  model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
  plot_model(model, to_file='modelcnn.png',show_shapes=True)
  return model
kfold = KFold(n_splits=4, shuffle=True, random_state=7)



tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 1), stop_words='english')


features = tfidf.fit_transform(df.content).toarray()

labels = df.category_id
features_train=features[train_indices]
labels_test=labels[test_indices]
features_test=features[test_indices]
labels_train=labels[train_indices]
Y = df.category.tolist()

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
features_train2=features[train_indices]
labels_test2=labels[test_indices]
features_test2=features[test_indices]
labels_train2=dummy_y[train_indices]


N = 3
kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

def baseline_model():
	# create model
	model = Sequential()
	n_words = features.shape[1]
	model.add(Dense(50, input_shape=(n_words,), activation='relu'))
	model.add(Dense(5, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	plot_model(model, to_file='modelffNN.png',show_shapes=True)
  
	return model

for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]


rfmodel=RandomForestClassifier(n_estimators=600, max_features='sqrt', max_depth=60, random_state=42)
rfmodel._name='RandomForest'
rfmodel.fit(features_train, labels_train)

nbmodel=MultinomialNB()
nbmodel._name='MultinomialNB'
nbmodel.fit(features_train, labels_train)


svModel=LinearSVC()
svModel._name='SVC'
svModel.fit(features_train, labels_train)

print("Testing results are as follows- ")
rfPred=rfmodel.predict(features_test)
print("Random forest classifier accuracy is ")
print(accuracy_score(labels_test, rfPred))
conf_mat_rf = confusion_matrix(labels_test, rfPred)
ax= sns.heatmap(conf_mat_rf, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(classification_report(labels_test, rfPred, target_names=category_id_df.category.values))
nbPred=nbmodel.predict(features_test)
print("Multinomial Naive Bayes classifier accuracy is ")
print(accuracy_score(labels_test, nbPred))
conf_mat_nb = confusion_matrix(labels_test, nbPred)
ax= sns.heatmap(conf_mat_nb, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(classification_report(labels_test, nbPred, target_names=category_id_df.category.values))
svcPred=svModel.predict(features_test)
print("SVC classifier accuracy is ")
print(accuracy_score(labels_test, svcPred))
conf_mat_sv = confusion_matrix(labels_test, svcPred)
ax= sns.heatmap(conf_mat_sv, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(classification_report(labels_test, svcPred, target_names=category_id_df.category.values))
nnmodel=KerasClassifier(build_fn=baseline_model, epochs=60, batch_size=10,verbose=0)
nnModel=baseline_model()
nnModel.fit(features_train2, train_labels_ff2,shuffle=True,epochs=60, batch_size=10,verbose=0)
nnmodel._name = 'ffNN'
ffPred=nnModel.predict_classes(features_test)



print("Feedforward accuracy is ")

print(accuracy_score(labels_test, ffPred))
conf_mat_ff = confusion_matrix(labels_test,ffPred )
ax= sns.heatmap(conf_mat_ff, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(classification_report(labels_test, ffPred, target_names=category_id_df.category.values))



cnnModel = KerasClassifier(build_fn=convModel, epochs=30,batch_size=32,verbose=0)
cnnModel.fit(train_data_ff2, train_labels_ff2,shuffle=True,epochs=30,batch_size=32,verbose=0)
cnnModel._name = 'ConvNN'
cnnPred=cnnModel.predict(test_data_ff2)
print("CNN model accuracy is ")
print(accuracy_score(labels_test2, cnnPred))
conf_mat_cnn = confusion_matrix(labels_test2, cnnPred)
ax= sns.heatmap(conf_mat_cnn, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(classification_report(labels_test2, cnnPred, target_names=category_id_df.category.values))