import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Tải xuống các gói dữ liệu cần thiết từ NLTK
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Chuyển văn bản thành chữ thường và tách từ
    tokens = text.lower().split()
    # Lọc các từ không phải là từ dừng (stopwords) và chỉ giữ lại các từ bao gồm chữ cái hoặc số
    filtered_words = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return ' '.join(filtered_words)  # Kết hợp lại các từ thành chuỗi văn bản

# Đọc dữ liệu từ file CSV
spam_df = pd.read_csv("sms_spam.csv", encoding='latin-1')

# Chuyển đổi nhãn từ 'spam' và 'ham' sang số (1 cho spam và 0 cho ham)
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Tiền xử lý các tin nhắn bằng hàm preprocess_text
spam_df['Message'] = spam_df['Message'].apply(preprocess_text)

# Chia dữ liệu thành tập huấn luyện và kiểm tra (75% huấn luyện, 25% kiểm tra)
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25, stratify=spam_df.spam, random_state=1)

# Khởi tạo CountVectorizer để chuyển đổi văn bản thành ma trận đặc trưng
cv = CountVectorizer()

# Chuyển đổi tập huấn luyện thành ma trận đếm
x_train_count = cv.fit_transform(x_train.values)

# Khởi tạo và huấn luyện mô hình Multinomial Naive Bayes
multinomial = MultinomialNB()
multinomial.fit(x_train_count, y_train)

# Chuyển đổi tập kiểm tra thành ma trận đếm
x_test_count = cv.transform(x_test)

# Dự đoán nhãn cho tập kiểm tra
y_pred = multinomial.predict(x_test_count)

# Tính toán các chỉ số đánh giá
accuracy_nb = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_multinomial = round(multinomial.score(x_train_count, y_train) * 100, 2)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# In ra ma trận nhầm lẫn và các chỉ số đánh giá
print('Confusion matrix for Naive Bayes\n', cm)
print('accuracy_Naive Bayes: %.3f' % accuracy)
print('precision_Naive Bayes: %.3f' % precision)
print('recall_Naive Bayes: %.3f' % recall)
print('f1-score_Naive Bayes: %.3f' % f1)
