import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    return pd.read_csv(url)

def extract_title(name):
    # Extract title like Mr, Mrs, etc.
    return name.split(',')[1].split('.')[0].strip()

def preprocess_data(df):
    df = df.drop(columns=['Cabin', 'Ticket', 'PassengerId'])

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Create new features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Title'] = df['Name'].apply(extract_title)

    # Simplify rare titles
    rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # One-hot encode 'Title'
    df = pd.get_dummies(df, columns=['Title'], drop_first=True)

    # Select final features
    df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize'] + 
            [col for col in df.columns if col.startswith('Title_')] + ['Survived']]

    # Normalize numeric columns
    scaler = StandardScaler()
    df[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(df[['Age', 'Fare', 'FamilySize']])

    return df
