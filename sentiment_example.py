from flask import Flask

import requests

app = Flask(__name__)


@app.route('/get_data_count', methods=['GET'])
def get_data_count():

   try:
       result_dict = {'negative' :0 , 'positive': 1}
       import psycopg2
       connection = psycopg2.connect(user="postgres", password="123456789", host="127.0.0.1", port="5432", database="employees_database")
       cursor = connection.cursor()
       label_name = input ("positive=1 & negative=0 ?")
       count = input ("Number of texts?")
       if (count is None):
           cursor.execute("SELECT COUNT(label_id) from data_labeling WHERE label_id = %s" , str(label_name))
           print ("Number of " + str(label_name) + " texts :")
           return cursor.fetchall()
       else:
           cursor.execute("SELECT COUNT(label_id) from data_labeling WHERE label_id = %s limit %s" , str(label_name), str(count))
           print ("Number of " + str(label_name) + " texts " + str(count)+":")
           return cursor.fetchall()
       cursor.close()
       connection.close()
   except:
       return "ERROR"

@app.route('/get_data', methods=['GET'])
def get_data():

   try:
       import psycopg2
       connection = psycopg2.connect(user="postgres", password="123456789", host="127.0.0.1", port="5432", database="employees_database")
       cursor = connection.cursor()
       result_dict = {'Descending' :"DESC" , 'Progressive': "ASC"}
       sort_order = input ("sort order Descending&Progressive?")
       count = input ("Number of texts?")
       cursor.execute("SELECT name_id, name_date,label_name,name from data_labeling INNER JOIN data_input ON name_id = id ORDER BY name_date %s limit %s" , result_dict[sort_order] ,str(count))
       return cursor.fetchall()
       cursor.close()
       connection.close()
   except:
       return "ERROR"



if __name__ == "__main__":
   app.run(debug=True, port=3000)

