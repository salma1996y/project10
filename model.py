
import numbers as np


import  glob

negative_files = glob.glob('Twitter/Negative/*.txt')
positive_files = glob.glob('Twitter/Positive/*.txt')


print("hh")

print(negative_files)
print("jjj")
print(positive_files)


def clean_text(text):
    from re import sub

    email = re.compile(r"\S*@\S*\s?")
    text = email.sub(' ', text) #delete email addresses
    text = re.sub('[^ ]+\.[^ ]+',' ',text) #delete links
    text = re.sub('[^A-Za-z0-9\']', ' ', text)
    text = text.lower()
    text=re.sub(r'[0-9]', ' ', text) #delete numbers
    text = re.sub(r"\s\s+",' ',text) #for remove extra spaces
    text = text.replace('\n', ' ').replace('\r', ' ') #delete new line
    #numbers = re.compile('[@!#$%^&*()<>?/\|}{~:]')
    #text = numbers.sub(' ', text) #delete special charachters
    return text

positive_texts = []
negative_texts = []

for file in positive_files:
    with open(file, 'r', encoding='utf-8') as file_to_read:
        try:
            text = file_to_read()
            text = clean_text(text)
            if text == "":
                continue
            print (text)
            positive_texts.append(text)
            print ("_"* 10)
        except UnicodeDecodeError:
            continue

print ("kkk")
print (len(negative_texts))
print ("sss")
print (len(positive_texts))

positive_labels = [1]*len(positive_texts)
negative_labels = [0]*len(negative_texts)

all_texts = positive_texts + negative_texts
all_labels = positive_labels + negative_labels

print("xxxx")
print (len(all_labels)==len(all_texts))



from sklearn.utils import shuffle
all_texts, all_labels = shuffle(all_texts, all_labels)

from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all_texts,all_labels, test_size=0.20)

from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
Vectorizer.fit(x_train)
x_train = Vectorizer.transform(X_train)

from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

x_test =Vectorizer.transform(x_test)

predictions = model.predict(x_test)

print ("cccc")
print (accuracy_score(y_test, predictions))

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print ("vvvv")
print (accuracy_score(y_test, predictions))



import pickle

with open('model.pickle', 'wb') as file:
    pickle.dump(model, file)

with open('Vectorizer.pickle', 'wb') as file:
    pickle.dump(Vectorizer,file)

with open('model.pickle', 'rb') as file:
    model = pickle.load(file)

with open('Vectorizer.pickle', 'wb') as file:
    Vectorizer = pickle.load(file)





example_test = 'bb '
cleaned_example_test = clean_text(example_test)
example_test_vector = Vectorizer.transform([cleaned_example_test])
example_result = model.predict(example_test_vector)
print ("zzz", example_test)
print (example_result(0))

