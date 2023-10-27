import pandas as pd
import numpy as np
import os
from etnn.data import DEFAULT_DATA_PATH


def prepare_1_ferris(
        df_name_input: str = "Sleep_health_and_lifestyle_dataset.csv",
        dataset_path: str = DEFAULT_DATA_PATH,
        df_name_output: str = 'health_dataset_preprocessed-1.csv',
        try_pregen: bool = True
) -> pd.DataFrame:
    # CHECK IF PRECOMPUTED
    if try_pregen and os.path.isfile(os.path.join(dataset_path, df_name_output)):
        return pd.read_csv(os.path.join(dataset_path, df_name_output))

    # LOADING
    # join path
    df_path = os.path.joing(dataset_path, df_name_input)

    # define column names (old ones unsuitable for d.col_name usage)
    columns = [
        'id', 'gender', 'age', 'occupation', 'sleep_duration', 'sleep_quality', 'physical_activity', 'stress_level',
        'bmi', 'blood_pressure', 'heart_rate', 'daily_steps', 'sleep_disorder'
    ]

    # load dataset
    df = pd.read_csv(df_path, names=columns, header=0)

    # REPLACE NAN VALUES
    df.sleep_disorder.fillna('No', inplace=True)

    # CONVERT TEXT FIELDS INTO INTEGER VALUES
    # bmi
    bmi_dict = {
        'Normal': 0,
        'Normal Weight': 1,
        'Overweight': 2,
        'Obese': 3
    }
    df.bmi = df.bmi.map(bmi_dict)

    # blood pressure
    df['blood_pressure1'] = df.blood_pressure.map(lambda x: int(x.split('/')[0]))
    df['blood_pressure2'] = df.blood_pressure.map(lambda x: int(x.split('/')[1]))
    df = df.drop("blood_pressure", axis=1)

    # occupation
    occ = df.occupation.unique()
    occ = dict(zip(occ, np.arange(len(occ)) + 1))
    df.occupation = df.occupation.map(occ)

    # sleep
    sleep_dict = {
        'No': 0,
        'Sleep Apnea': 1,
        'Insomnia': 2
    }
    df.sleep_disorder = df.sleep_disorder.map(sleep_dict)

    # gender
    df['gender_male'] = df.gender.map(lambda x: x == 'Male')
    df['gender_female'] = df.gender.map(lambda x: x == 'Female')
    df['gender_other'] = df.gender_female == df.gender_male
    df = df.drop('gender', axis=1)

    df.to_csv(os.path.join(dataset_path, df_name_output), index=False)

    return df



