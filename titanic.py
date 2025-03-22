from sklearn.ensemble import RandomForestClassifier
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1500)

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
gender_data = pd.read_csv('data/gender_submission.csv')

# plt.plot()
# sns.countplot(x='Survived', hue='Sex', data=train_data)
# sns.countplot(x='Survived', hue='Pclass', data=train_data)
# plt.show()

features = ['Sex', 'Pclass']

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
y = train_data['Survived']

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)
model.fit(X, y)
prediction = model.predict(X_test)
output_df = pd.DataFrame({'PassengerID' : test_data.PassengerId, 'Survived' : prediction})
print(output_df.head(50))