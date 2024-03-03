import pandas as pd
import selenium.webdriver as wd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


driver_path="C:/Users/Admin/Desktop/pythonpractice/chromedriver.exe"
driver=wd.Chrome(executable_path=driver_path)
driver.get("https://www.worldometers.info/coronavirus/")
wait=WebDriverWait(driver,10)
target_tr=wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR,"table#main_table_countries_today tbody tr")))
list_country_deaths_recovered=[]
def remove_commas(number_in_string_received):
    if len(number_in_string_received)==0 or number_in_string_received=="N/A":
        return number_in_string_received
    number_string=""
    for i in number_in_string_received:
        if i==",":
            continue
        else:
            number_string=number_string+i
    return int(number_string)


for i in target_tr:
    target_tds=i.find_elements(By.TAG_NAME,"td")
    specific_td_country=target_tds[1]
    country=specific_td_country.text
    td_total_cases=target_tds[2]
    total_cases=remove_commas(td_total_cases.text)
    td_total_deaths=target_tds[4]
    total_deaths=remove_commas(td_total_deaths.text)
    td_total_recovered=target_tds[6]
    total_recovered=remove_commas(td_total_recovered.text)
    list_country_deaths_recovered.append([country,total_cases,total_deaths,total_recovered])
data_frame=pd.DataFrame(list_country_deaths_recovered,columns=["country","total_cases","deaths","recovered"])
data_frame=data_frame.fillna("hello")
print(data_frame)
data_frame=data_frame.drop([0,246],axis=0)
info=data_frame.info()
data_frame.to_excel("covid_selenium.xlsx")

print(data_frame)
print(info)

data_frame=pd.read_excel("covid_selenium.xlsx")
data_frame=data_frame.dropna()
text_based_features=data_frame["country"]
vectorizer=TfidfVectorizer(stop_words="english",max_features=5000)
text_based_features_vectorized=vectorizer.fit_transform(text_based_features)
numerical_features=data_frame.drop(["country"],axis=1)
combined_text_numerical=hstack([text_based_features_vectorized,numerical_features])
number_of_clusters=5
model=KMeans(n_clusters=number_of_clusters,random_state=42)
predicted_clusters=model.fit_predict(combined_text_numerical)
data_frame["clusters"]=predicted_clusters
print(data_frame)
data_frame.to_excel("covid_selenium.xlsx")

data_frame=pd.read_excel("covid_selenium.xlsx")
text_based_features=data_frame["country"]
vectorizer=TfidfVectorizer(stop_words="english",max_features=5000)
text_based_features_vectorized=vectorizer.fit_transform(text_based_features)
numerical_features=data_frame.drop(["country","clusters"],axis=1)
combined_text_numerical=hstack([text_based_features_vectorized,numerical_features])
X=combined_text_numerical
Y=data_frame["clusters"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(X_train,Y_train)
joblib.dump(model,"covid_classification.joblib")
print("model has been trained")
loaded_model=joblib.load("covid_classification.joblib")
prediction=loaded_model.predict(X_test)
accuracy_score=accuracy_score(Y_test,prediction)
print(accuracy_score)
dummy_data=pd.DataFrame([["Pakistan",110309712,120000,300000000]],columns=["city","cases","deaths","recovered"])
text_based_features_aa=dummy_data["city"]
text_based_features_aa_vectorized=vectorizer.transform(text_based_features_aa)
numerical_features_aa=dummy_data.drop(["city"],axis=1)
combined_aa=hstack([text_based_features_aa_vectorized,numerical_features_aa])
prediction_aa=loaded_model.predict(combined_aa)
print(prediction_aa)


import pandas as pd
import selenium.webdriver as wd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

driver_path="C:/Users/Admin/Desktop/pythonpractice/chromedriver.exe"
driver=wd.Chrome(executable_path=driver_path)
driver.get("https://www.worldometers.info/coronavirus/")
wait=WebDriverWait(driver,10)
target_tr=wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR,"table#main_table_countries_today tbody tr")))
list_country_cases_deaths_recovered=[]
def remove_commas(number_string_received):
    if len(number_string_received)==0 or number_string_received=="N/A":
        return number_string_received
    else:
        new_number=""
        for i in number_string_received:
            if i==",":
                continue
            else:
                new_number=new_number+i
        return new_number
for i in target_tr:
    target_tds=i.find_elements(By.TAG_NAME,"td")
    specific_td=target_tds[1]
    country=specific_td.text
    td_total_cases=target_tds[2]
    total_cases=remove_commas(td_total_cases.text)
    td_total_deaths=target_tds[4]
    total_deaths=remove_commas(td_total_deaths.text)
    td_recovered=target_tds[6]
    recovered=remove_commas(td_recovered.text)
    list_country_cases_deaths_recovered.append([country,total_cases,total_deaths,recovered])
data_frame=pd.DataFrame(list_country_cases_deaths_recovered,columns=["country","cases","deaths","recoveries"])
data_frame=data_frame.drop([0,246],axis=0)
data_frame.to_excel("covid_selenium.xlsx")

data_frame=pd.read_excel("covid_selenium.xlsx")
data_frame["recoveries"]=data_frame["recoveries"].replace("N/A",pd.NA)
data_frame["recoveries"]=data_frame["recoveries"].fillna(0)
data_frame=data_frame.dropna()
text_based_features=data_frame["country"]
vectorizer=TfidfVectorizer(stop_words="english",max_features=5000)
text_based_features_vectorized=vectorizer.fit_transform(text_based_features)
numerical_features=data_frame.drop(["country"],axis=1)
combined_text_numerical=hstack([text_based_features_vectorized,numerical_features])
number_of_clusters=5
model=KMeans(n_clusters=number_of_clusters,random_state=42)
predicted_clusters=model.fit_predict(combined_text_numerical)
data_frame["clusters"]=predicted_clusters
data_frame.to_excel("covid_selenium.xlsx")
print(data_frame)

data_frame=pd.read_excel("covid_selenium.xlsx")
text_based_features=data_frame["country"]
vectorizer=TfidfVectorizer(stop_words="english",max_features=5000)
text_based_features_vectorized=vectorizer.fit_transform(text_based_features)
numerical_features=data_frame.drop(["country","clusters"],axis=1)
combined_text_numerical=hstack([text_based_features_vectorized,numerical_features])
X=combined_text_numerical
Y=data_frame["clusters"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(X_train,Y_train)
joblib.dump(model,"covid_classification.joblib")
print("model has been trained")
loaded_model=joblib.load("covid_classification.joblib")
prediction=loaded_model.predict(X_test)
accuracy_score=accuracy_score(Y_test,prediction)
print(accuracy_score)
dummy_data=pd.DataFrame([["Pakistan",1955368423,10000000,123]],columns=["country","cases","deaths","recoveries"])
text_based_features_aa=dummy_data["country"]
text_based_features_aa_vectorized=vectorizer.transform(text_based_features_aa)
numerical_features_aa=dummy_data.drop(["country"],axis=1)
combined_t_n=hstack([text_based_features_aa_vectorized,numerical_features_aa])
prediction=loaded_model.predict(combined_t_n)
print(prediction)

