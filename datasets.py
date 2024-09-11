import pandas

pandas.__version__

from sklearn import datasets

import pandas as pd

# Load the diabetes dataset and create a dataframe

diabetes = datasets.load_diabetes()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Add the target variable to the dataframe

df['target'] = diabetes.target

# Print the first 5 rows of some variables

print(df[['s1', 'sex', 'bmi', 'target']].head())


corr = df[['s1', 'age', 'bmi', 'target']].corr()

print(corr)

'''As seen above, we can clearly make a conclusion that the higher the total
serum cholesterol (s1), the higher the chance of someone contracting diabetes.
In this way, total serum cholesterol is positively correlated with the chance of
contracting diabetes.'''
