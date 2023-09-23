# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
REG NO:212222230156
NAME:S.M.Syed Mokthiyar

#import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#read the datset
df=pd.read_csv("/content/Churn_Modelling.csv")
print(df)

#dropout unwanted columns
df1=df.drop(['RowNumber','Age','Geography','Surname','Gender'],axis=1)
df1

#checking for null values
print(df1.isnull().sum())
df1.fillna(df.mean().round(1),inplace=True)
print(df1.duplicated())

#normalize the data
scalar=MinMaxScaler()
df2=pd.DataFrame(scalar.fit_transform(df1))
df2

#split the datset as x and y
x=df2.iloc[:,:-1].values
print(x)
y=df2.iloc[:,-1].values
print(y)

#split the dataset for training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_test)
print(x_train)
print(len(x_train))
print(len(x_test))
```

## OUTPUT:
# dataset
![dataset](https://github.com/syedmokthiyar/Ex.No.1---Data-Preprocessing/assets/118787294/a37adaaf-5069-4125-8408-0024bc8b56e2)

# checking for null values
![check  exp 1 nn](https://github.com/syedmokthiyar/Ex.No.1---Data-Preprocessing/assets/118787294/16f46308-e456-4187-bc2c-47c91d8bdab8)

# normalize the data
![exp 1 nn norm](https://github.com/syedmokthiyar/Ex.No.1---Data-Preprocessing/assets/118787294/1e632be7-71c9-4279-a63f-8269e221d7fd)

# split the datset as x and y
![spilt data exp 1nn](https://github.com/syedmokthiyar/Ex.No.1---Data-Preprocessing/assets/118787294/36fccd32-b944-4480-a034-a02aeecfd256)

# split the dataset for training and testing
![nn exp 1 l](https://github.com/syedmokthiyar/Ex.No.1---Data-Preprocessing/assets/118787294/c304fa55-264c-4690-91d0-d37eec1a686f)


## RESULT
Hence, the Data preprocessing is performed over a data set successfully.
