from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

df = pd.read_csv (r'Dataset_IC.csv')

X = df.iloc[:, 0:5].values
y = df.iloc[:, 5].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
sc.fit(X_train)
with open('stand_scalar','wb') as f:
  pickle.dump(sc,f)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = RandomForestClassifier(n_estimators=100,
bootstrap = True,
max_features = 'sqrt')

ppn.fit(X_train_std, y_train)

with open('model','wb') as f:
  pickle.dump(ppn,f)