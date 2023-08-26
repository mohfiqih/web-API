from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Muat data dari CSV
data_clean = pd.read_csv('model/dataclean_TA.csv', encoding='latin1')

# Set label sebagai tipe data kategori dan ulasan sebagai tipe data string
data_clean = data_clean.astype({'label': 'category'})
data_clean = data_clean.astype({'ulasan': 'string'})

# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()
text_tf = tf.fit_transform(data_clean['ulasan'].astype('U'))

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(text_tf, data_clean['label'], test_size=0.20, random_state=42)

# Inisialisasi dan latih model Naive Bayes
NaiveBayes = MultinomialNB().fit(X_train, y_train)

# Lakukan prediksi pada data uji
predicted = NaiveBayes.predict(X_test)

# Dapatkan daftar label unik yang ada dalam data uji
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

    # Masukkan hasil TP, FN, TN, FP ke dalam dictionary
    evaluation_by_label[label] = {'TP': TP, 'FN': FN, 'TN': TN, 'FP': FP}

# Cetak hasil evaluasi untuk setiap label
for label, eval_dict in evaluation_by_label.items():
    print(f"\nEvaluasi untuk label '{label}':")
    print("True Positive (TP):", eval_dict['TP'])
    print("False Negative (FN):", eval_dict['FN'])
    print("True Negative (TN):", eval_dict['TN'])
    print("False Positive (FP):", eval_dict['FP'])

# Hitung metrik evaluasi keseluruhan
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, average="weighted")
recall = recall_score(y_test, predicted, average="weighted")
f1score = f1_score(y_test, predicted, average="weighted")

# Cetak hasil evaluasi keseluruhan
print("\nHasil Evaluasi Keseluruhan:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1score)

# Cetak laporan klasifikasi untuk semua kelas
print("\nEvaluasi Keseluruhan:")
print(classification_report(y_test, predicted))
