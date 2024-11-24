import numpy as np
from scipy.io import arff
import pandas as pd

def label(fp):
    data = pd.read_csv(fp)
    target = data['Class']
    data = data.drop(columns=['Class'])
    return data, target
def main():
    data, meta = arff.loadarff('Obesity_Dataset.arff')
    df = pd.DataFrame(data)
    df.rename(columns={'Overweight_Obese_Family': 'Family',
                       'Consumption_of_Fast_Food': 'Junk',
                       'Frequency_of_Consuming_Vegetables': 'Vegetables',
                       'Number_of_Main_Meals_Daily': 'Meals',
                       'Food_Intake_Between_Meals': 'Between',
                       'Liquid_Intake_Daily': 'Liquids',
                       'Calculation_of_Calorie_Intake': 'Calories',
                       'Physical_Excercise' : 'Exercise',
                       'Schedule_Dedicated_to_Technology' : 'Technology', 
                       'Type_of_Transportation_Used' : 'Transportation'
                       }, inplace=True)
    df = df.astype(int)

    # binary values to 0, 1 (saved as 1, 2)
    df['Sex'] -= 1
    df['Family'] -= 1
    df.to_csv('obesity.csv', index=False)
    df['Age_scaled'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
    df['Height_scaled'] = (df['Height'] - df['Height'].min()) / (df['Height'].max() - df['Height'].min())
    df.to_csv('obesity_min_max_scaled.csv', index=False)
    # for KNN we should not drop variables. One-hot with drop and not one-hotting at all
    # doesnt change binary variables, but one-hotting with no drop will double the distance
    df = pd.get_dummies(df, columns=['Transportation', 'Sex', 'Family', 'Junk', 'Smoking', 'Calories'])
    df.to_csv('obesity_encoded.csv', index=False)
    df = pd.get_dummies(df, columns=['Vegetables', 'Meals', 'Between', 'Liquids', 'Exercise', 'Technology'])
    df.to_csv('obesity_neural.csv', index=False)
    df['Class'] -= 1
    print('complete')

if __name__ == '__main__':
    main()