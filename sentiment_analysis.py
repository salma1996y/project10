import requests


import pickle

#with open('model.pickle', 'rb') as file:
#    model = pickle.load(file)

#with open('vectorizer.pickle', 'rb') as file:
#    vectorizer = pickle.load(file)

database_service =  request.get("http://127.0.0.1:3000" +"/get_data_count" , params={'label_name': 0 , 'count': 10})

import re
def clean_text(text):
    text = text.lower()
    text = re.sub("@[a-z0-9_]+", ' ', text)
    text = re.sub("[^ ]+\.[^ ]+", ' ', text)
    text = re.sub("[^ ]+@[^ ]+\.[^ ]", ' ', text)
    text = re.sub("[^a-z\' ]", ' ', text)
    text = re.sub(' +', ' ', text)

    return text
database_service = database_service.json()

cleaned_test = clean_text(database_service)

test_vector = vectorizer.transform([cleaned_test])

result = model.predict(testـvector)

print("Sentence classification:", database_service)

print(result[1000])

