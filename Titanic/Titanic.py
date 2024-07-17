import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# データの読み込み
print("データの読み込み中...")
train = pd.read_csv(r'C:\Users\etatyos\OneDrive - Ericsson\Desktop\Kaggle\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\etatyos\OneDrive - Ericsson\Desktop\Kaggle\Titanic\test.csv')
print("データの読み込み完了")

# 欠損値の補完
print("欠損値の補完中...")
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
print("欠損値の補完完了")

# 特徴量エンジニアリングの追加
print("特徴量エンジニアリング中...")

# Title の抽出
for dataset in [train, test]:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Title のマッピング
title_mapping = {
    "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Col": 7, "Major": 7, "Mlle": 8, "Countess": 9,
    "Ms": 2, "Lady": 9, "Jonkheer": 9, "Don": 10, "Dona": 10, "Mme": 2, "Capt": 7, "Sir": 10
}
for dataset in [train, test]:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# FamilySize の作成
for dataset in [train, test]:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# IsAlone の作成
for dataset in [train, test]:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print("特徴量エンジニアリング完了")

# カテゴリカルデータのエンコーディング
print("カテゴリカルデータのエンコーディング中...")
label_encoder = LabelEncoder()
train['Sex'] = label_encoder.fit_transform(train['Sex'])
test['Sex'] = label_encoder.transform(test['Sex'])

train['Embarked'] = label_encoder.fit_transform(train['Embarked'])
test['Embarked'] = label_encoder.transform(test['Embarked'])
print("カテゴリカルデータのエンコーディング完了")

# 特徴量とターゲットの設定
print("特徴量とターゲットの設定中...")
features = ['Pclass', 'Age', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Title', 'IsAlone']
X = train[features]
y = train['Survived']
print("特徴量とターゲットの設定完了")

# データの分割
print("データの分割中...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("データの分割完了")

# ランダムフォレストモデルのハイパーパラメータチューニング
print("ハイパーパラメータチューニング中...")
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("ハイパーパラメータチューニング完了")

# 最適なモデルでの予測
print("最適なモデルでの予測中...")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"検証データセットの正確度: {accuracy}")

# テストデータでの予測
print("テストデータでの予測中...")
X_test = test[features]
test['Survived'] = best_model.predict(X_test)
print("テストデータでの予測完了")

# 現在の日時を取得してファイル名に追加
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'submission_optimized_{current_time}.csv'

# 提出用ファイルの作成
print("提出用ファイルの作成中...")
submission = test[['PassengerId', 'Survived']]
submission.to_csv(rf'C:\Users\etatyos\OneDrive - Ericsson\Desktop\Kaggle\Titanic\{filename}', index=False)
print(f"提出用ファイルの作成完了: {filename}")

# 結果の表示
print("最終結果の表示:")
print(accuracy, submission.head())
