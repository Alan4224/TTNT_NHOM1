import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

spam_df = pd.read_csv("sms_spam.csv")
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
x_train,x_test,y_train,y_test = train_test_split(spam_df.Message,spam_df.spam,test_size=0.25, stratify=spam_df.spam , random_state=1)
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)
multinomial = MultinomialNB()
multinomial.fit(x_train_count, y_train)
x_test_count = cv.transform(x_test)
y_pred = multinomial.predict(x_test_count)
accuracy_nb = round(accuracy_score(y_test,y_pred)*100,2)
acc_multinomial = round(multinomial.score(x_train_count,y_train)*100,2)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred,average='binary')
recall = recall_score(y_test,y_pred,average='binary')
f1 = f1_score(y_test,y_pred,average='binary')
print('Confusion matrix for Naive Bayes\n',cm)
print('accuracy_Naive Bayes: %.3f' %accuracy)
print('precision_Naive Bayes: %.3f' %precision)
print('recall_Naive Bayes: %.3f' %recall)
print('f1-score_Naive Bayes: %.3f' %f1)