#Liver Prediction

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Importing the dataset
dataset = pd.read_csv('ILPD.csv')

#Checking for null values
dataset.isnull().sum()

#Data Visualization

#(Visualizing Results of Diagnosed and Not Diagnosed Patients)
sns.countplot(data = dataset, x = 'Result', label='Count')

LD, NLD = dataset['Result'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)

#(Visualizing Results of Male and Not Female Patients)
sns.countplot(data = dataset, x = 'Gender', label='Count')

M, F = dataset['Gender'].value_counts()
print('Number of Male patients: ',M)
print('Number of Female patients: ',F)

#Visualizing if "Age" is a factor
sns.factorplot(x="Age", y="Gender", hue="Result", data= dataset);
# Therefore, we get to know that "AGE" IS A FACTOR

#Visualizing 3 columns and Male and Female count
dataset[['Gender', 'Result','Age']].groupby(['Result','Gender'], as_index=False).count().sort_values(by='Result', ascending=False)

#Visualizing 3 columns and Male and Female Average Age
dataset[['Gender', 'Result','Age']].groupby(['Result','Gender'], as_index=False).mean().sort_values(by='Result', ascending=False)

#Visualing using "FacetGrid" diagram between "Age" and "Result"
g = sns.FacetGrid(dataset, col="Result", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');

#Visualing using "FacetGrid" diagram between "Bilirubin Levels" and "Total Bilirubin Levels"
g = sns.FacetGrid(dataset, col="Gender", row="Result", margin_titles=True)
g.map(plt.scatter, "Direct Bulirubin", "Total Bilirubin", edgecolor = 'w')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Total Bilirubin and Bilirubin Comparison');

#There seems to be direct relationship between Total_Bilirubin and Direct_Bilirubin. We have the possibility of removing one of this feature.

sns.jointplot("Total Bilirubin", "Direct Bulirubin", data= dataset, kind="reg")

#Visualing using "FacetGrid" diagram between "Aspartate Aminotransferase" and "Alamine Aminotransferase Levels" 
g = sns.FacetGrid(dataset, col="Gender", row="Result", margin_titles=True)
g.map(plt.scatter, "Aspartate Aminotransferase", "Alamine Aminotransferase", edgecolor = 'w')
plt.subplots_adjust(top=0.9)

#There seems to be linear relationship between TAspartate Aminotransferase and Alamine Aminotransferase. We have the possibility of removing one of this feature.

sns.jointplot("Aspartate Aminotransferase", "Alamine Aminotransferase", data= dataset, kind="reg")

#Visualing using "FacetGrid" diagram between "Alkaline Phosphotase" and "Alamine Aminotransferase Levels" 
g = sns.FacetGrid(dataset, col="Gender", row="Result", margin_titles=True)
g.map(plt.scatter, "Alkaline Phosphotase", "Alamine Aminotransferase", edgecolor = 'w')
plt.subplots_adjust(top=0.9)

#There No linear correlation between Alkaline Phosphotase and Alamine Aminotransferase

#Visualing using "FacetGrid" diagram between "Total Proteins" and "Albumin" Levels" 
g = sns.FacetGrid(dataset, col="Gender", row="Result", margin_titles=True)
g.map(plt.scatter, "Total Protiens", "Albumin", edgecolor = 'w')
plt.subplots_adjust(top=0.9)

'''There is linear relationship between Total Protiens and Albumin and the gender. We have the possibility of removing one of this feature.'''

sns.jointplot("Total Protiens", "Albumin", data=dataset, kind="reg")

#Visualing using "FacetGrid" diagram between "Total Proteins" and "Albumin" Levels" 
g = sns.FacetGrid(dataset, col="Gender", row="Result", margin_titles=True)
g.map(plt.scatter, "Albumin", "Albumin and Globulin Ratio", edgecolor = 'w')
plt.subplots_adjust(top=0.9)

'''There is linear relationship between Albumin_and_Globulin_Ratio and Albumin. We have the possibility of removing one of this feature.'''

sns.jointplot("Albumin", "Albumin and Globulin Ratio", data=dataset, kind="reg")

'''From the above jointplots and scatterplots, we find direct relationship between the following features:

1. Direct_Bilirubin & Total_Bilirubin
2. Aspartate_Aminotransferase & Alamine_Aminotransferase
3. Total_Protiens & Albumin
4. Albumin_and_Globulin_Ratio & Albumin'''
    
''' Hence, we can very well find that we can omit one of the features. I'm going to keep the follwing features:
        
1. Total_Bilirubin
2. Alamine_Aminotransferase
3. Total_Protiens
4. Albumin_and_Globulin_Ratio
5. Albumin '''

# Convert categorical variable "Gender" to indicator variables
pd.get_dummies(dataset['Gender'], prefix = 'Gender').head()

# Add the dummy variables to the dataset using the CONCAT function.
dataset = pd.concat([dataset ,pd.get_dummies(dataset['Gender'], prefix = 'Gender')], axis=1)

# Fill missing data with mean values
dataset['Albumin and Globulin Ratio'] = dataset["Albumin and Globulin Ratio"].fillna(dataset['Albumin and Globulin Ratio'].mean())

# Defining matrices (Depdendent and Independent) 
X = dataset.drop(['Gender','Result'], axis=1)
Y = dataset['Result']

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 101 )

#Figure out correlation
liver_corr = X.corr()

#Building a heat map

plt.figure(figsize = (10, 10))
sns.heatmap(data = liver_corr, cmap = "coolwarm", cbar = True,  square = True, fmt = '.2f', annot=True, annot_kws={'size': 15})
plt.title('Correlation amongst features')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''--------------------------------------------------------------------------------------------------------------------------------'''
# Machine Learning #

#SVM
# Fitting the SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('white', 'black'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.show()

