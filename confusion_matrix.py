from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import re
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np

# -------------------------  -----------------------------------#
data = pd.read_csv('model/dataset_skripsi.csv', sep=",", encoding="UTF-8")
#casefolding
def casefolding(ulasan):
  ulasan = ulasan.lower()
  ulasan = ulasan.strip(" ")
  ulasan = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|
\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))\n''', '', ulasan)
  return ulasan
data['ulasan'] = data['ulasan'].apply(casefolding)
data.head(10000)

#tokenizing
def token(ulasan):
  nstr = ulasan.split(' ')
  dat = []
  a = -1
  for hu in nstr:
    a = a + 1
    if hu == '':
      dat.append(a)
    p = 0
    b = 0
    for q in dat:
        b = q - p
        del nstr[b]
        p = p + 1
    return nstr
data['ulasan'] = data['ulasan'].apply(token)
data.head(10000)

#filtering kata
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def stopword_removal(ulasan):
  filtering = stopwords.words('indonesian')
  x = []
  data = []
  def myfunc(x):
    if x in filtering:
      return False
    else:
      return True
  fit = filter(myfunc, ulasan)
  for x in fit:
    data.append(x)
  return data
data['ulasan'] = data['ulasan'].apply(stopword_removal)
data.head(10000)


#  ----------------------------- DATA Clean --------------------------- #
data_clean = pd.read_csv('model/dataclean_TA.csv', encoding='latin1')
data_clean = data_clean.astype({'label': 'category'})
data_clean = data_clean.astype({'ulasan': 'string'})

#  ----------------------------- Tokenizer --------------------------- #
#Pembuatan Kamus kata
t  = Tokenizer()
fit_text = data['ulasan']
t.fit_on_texts(fit_text)

#Pembuatan Id masing-masing kata
sequences = t.texts_to_sequences(data['ulasan'])

#hapus duplikat kata yang muncul
list_set_sequence = [list(dict.fromkeys(seq)) for seq in sequences]

#mencari max length sequence
def FindMaxLength(lst):
    maxList = max((x) for x in lst)
    maxLength = max(len(x) for x in lst )
    return maxList, maxLength

# Driver Code
max_seq, max_length_seq = FindMaxLength(list_set_sequence)
jumlah_index = len(t.word_index) +1

count_word = [len(i) for i in list_set_sequence]
# print('list panjang kalimat : ', count_word)
max_len_word = max(count_word)
# print(max_len_word)

#  ----------------------------- CountVector --------------------------- #
min_len_word=min(count_word)
# print(min_len_word)
# print ("Original list is : " + str(list_set_sequence[0]))

#Padding
from keras_preprocessing.sequence import pad_sequences

padding= pad_sequences([list(list_set_sequence[i]) for i in range(len(list_set_sequence))],
maxlen= max_len_word, padding='pre')
padding[:10]

reviews2 = [" ".join(r) for r in data['ulasan']]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(reviews2)
# print(vectorizer.vocabulary_ )
# print(X_vec.todense())

set_of_words = set()
for sentence in reviews2:
    for word in sentence.split():
        set_of_words.add(word)
vocab = list(set_of_words)
# print(vocab)

position = {}
for i, token in enumerate(vocab):
    position[token] = i
# print(position)

bow_matrix = np.zeros((len(reviews2), len(vocab)))

for i, preprocessed_sentence in enumerate(reviews2):
    for token in preprocessed_sentence.split():
        bow_matrix[i][position[token]] = bow_matrix[i][position[token]] + 1

bow_matrix

vectorizer_ngram_range = CountVectorizer(analyzer='word', ngram_range=(1,3))
bow_matrix_ngram = vectorizer_ngram_range.fit_transform(reviews2)

# print(vectorizer_ngram_range)
# print(bow_matrix_ngram.toarray())

# Max Feature
vectorizer_max_features = CountVectorizer(analyzer='word', ngram_range=(1,3), max_features = 100)
bow_matrix_max_features = vectorizer_max_features.fit_transform(reviews2)

# print(vectorizer_max_features)
# print(bow_matrix_max_features.toarray())

vectorizer_max_min = CountVectorizer(analyzer='word', ngram_range=(1,3), max_df =3, min_df = 2)
bow_matrix_max_min = vectorizer_max_min.fit_transform(reviews2)

# print(vectorizer_max_min)
# print(bow_matrix_max_min.toarray())

# TF IDF
vectorizer = TfidfVectorizer()
tf_idf_matrix = vectorizer.fit_transform(reviews2)

# print(vectorizer)
# print(tf_idf_matrix.toarray())
# print("\nThe shape of the TF-IDF matrix is: ", tf_idf_matrix.shape)

vectorizer_l1_norm = TfidfVectorizer(norm="l1")
tf_idf_matrix_l1_norm = vectorizer_l1_norm.fit_transform(reviews2)

# print(vectorizer_l1_norm)
# print(tf_idf_matrix_l1_norm.toarray())
# print("\nThe shape of the TF-IDF matrix is: ", tf_idf_matrix_l1_norm.shape)

vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2", analyzer='word', ngram_range=(1,3), max_features = 6)
tf_idf_matrix_n_gram_max_features = vectorizer_n_gram_max_features.fit_transform(reviews2)

# print(vectorizer_n_gram_max_features)
# print(tf_idf_matrix_n_gram_max_features.toarray())
# print("\nThe shape of the TF-IDF matrix is: ", tf_idf_matrix_n_gram_max_features.shape)

def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.dot(vector1, vector2) / (np.sqrt(np.sum(vector1 ** 2)) * np.sqrt(np.sum(vector2 ** 2)))
    
#  ------------------------- #
data_clean = pd.read_csv('model/dataclean_TA.csv', encoding='latin1')
data_clean.head(10000)

# Clean Data
data_clean = data_clean.astype({'label' : 'category'})
data_clean = data_clean.astype({'ulasan' : 'string'})
data_clean.dtypes


X=X_vec
y=data_clean['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=19)


NaiveBayes = MultinomialNB().fit(X_train, y_train)
predicted = NaiveBayes.predict(X_test)
X_train.shape

# Hitung confusion matrix
conf_matrix = confusion_matrix(y_test, predicted)
# Mendapatkan nilai TP, FN, TN, dan FP berdasarkan confusion matrix
TP_all = conf_matrix[1][1]  # True Positive untuk label 'Kurang'
FN_all = conf_matrix[1][0]  # False Negative untuk label 'Kurang'
TN_all = conf_matrix[0][0]  # True Negative untuk label 'Baik'
FP_all = conf_matrix[0][1]  # False Positive untuk label 'Baik'

print("Confusion Matrix:")
print(conf_matrix)
print("\nHasil Evaluasi:")
print("True Positive (TP):", TP_all)
print("False Negative (FN):", FN_all)
print("True Negative (TN):", TN_all)
print("False Positive (FP):", FP_all)

# ---------------------------- Conf Matrix Setiap Label ----------------------- #

unique_labels = data_clean['label'].unique()
# Inisialisasi dictionary untuk menyimpan hasil TP, FN, TN, FP untuk setiap label
evaluation_by_label = {}
# Hitung confusion matrix untuk setiap label
for label in unique_labels:
    conf_matrix = confusion_matrix(y_test, predicted, labels=[label, unique_labels[0]])
    TP = conf_matrix[0][0]
    FN = conf_matrix[0][1]
    TN = conf_matrix[1][1]
    FP = conf_matrix[1][0]
    evaluation_by_label[label] = {'TP': TP, 'FN': FN, 'TN': TN, 'FP': FP}

# Cetak hasil evaluasi untuk setiap label
for label, eval_dict in evaluation_by_label.items():
    print(f"\nEvaluasi untuk label '{label}':")
    print("True Positive (TP):", eval_dict['TP'])
    print("False Negative (FN):", eval_dict['FN'])
    print("True Negative (TN):", eval_dict['TN'])
    print("False Positive (FP):", eval_dict['FP'])

# Hitung metrik evaluasi keseluruhan
accuracy = accuracy_score(y_test, predicted)*100
precision = precision_score(y_test, predicted, average="weighted")
recall = recall_score(y_test, predicted, average="weighted")
f1score = f1_score(y_test, predicted, average="weighted")

print("\nHasil Evaluasi Keseluruhan:")
print("Accuracy:", accuracy)
print("Error:", 100 - accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1score)

print("\nEvaluasi Keseluruhan:")
print(classification_report(y_test, predicted))