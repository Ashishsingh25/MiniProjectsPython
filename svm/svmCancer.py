import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

# print(df_cancer.head())
# print(df_cancer.shape)
# print(df_cancer.columns)

df_cancerMean = df_cancer[df_cancer.columns[:10]]
df_cancerMean = df_cancer.iloc[:, list(range(10)) + [-1]]
# print(df_cancer.columns)

# sns.pairplot(df_cancerMean, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter','mean area','mean smoothness', 'mean compactness', 'mean concavity',
#        'mean concave points', 'mean symmetry', 'mean fractal dimension'] )
# plt.show()

# plt.figure(figsize=(20,12))
# sns.heatmap(df_cancer.corr(), annot=True)
# plt.show()

X = df_cancerMean.drop(['target'], axis = 1)
# X.head()

y = df_cancerMean['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min())
X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min())

from sklearn.svm import SVC
svc_model = SVC()
# print(svc_model)
svc_model.fit(X_train_scaled, y_train)
y_predict = svc_model.predict(X_test_scaled)

from sklearn.metrics import classification_report, confusion_matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
# print(confusion)
sns.heatmap(confusion, annot=True)
plt.show()
print(classification_report(y_test, y_predict))

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)

print (grid.best_params_)
print ('\n')
print (grid.best_estimator_)

from sklearn.metrics import classification_report, confusion_matrix
grid_predictions = grid.predict(X_test_scaled)
cm = np.array(confusion_matrix(y_test, grid_predictions, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])

sns.heatmap(confusion, annot=True)
plt.show()
print(classification_report(y_test, grid_predictions))




