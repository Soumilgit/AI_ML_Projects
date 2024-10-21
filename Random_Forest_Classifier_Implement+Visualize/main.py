import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


file_path1 = 'Health_subject1.csv'
file_path2 = 'Health_subject2.csv'

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)


df = pd.concat([df1, df2], ignore_index=True)

df_cleaned = df.dropna(subset=['Label'])
X = df_cleaned.drop('Label', axis=1).fillna(df_cleaned.mean())
y = df_cleaned['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(y), yticklabels=set(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10,7))
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()
