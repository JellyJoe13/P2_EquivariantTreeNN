import pandas as pd
import numpy as np
import os
from etnn.data import DEFAULT_DATA_PATH
from tqdm import tqdm
from etnn.data.ferris_score_helpers import build_wheel_happyness
import typing


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


def generate_ferris_dataset(
        num_gondolas: int = 10,
        num_part_pg: int = 5,
        num_to_generate: int = 1_000,
        df_name_input: str = "Sleep_health_and_lifestyle_dataset.csv",
        dataset_path: str = DEFAULT_DATA_PATH,
        df_intermediate_output_name: str = 'health_dataset_preprocessed-1.csv',
        try_pregen: bool = True,
        seed: int = 4651431
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    # set seed if not none
    if seed is not None:
        np.random.seed(seed)

    # load dataset which is the base of the data generation
    df_health = prepare_1_ferris(
        df_name_input=df_name_input,
        dataset_path=dataset_path,
        df_name_output=df_intermediate_output_name,
        try_pregen=try_pregen
    )

    # see if file already exists and load this
    # file name logic
    file_name = f"ferris-wheel_g-{num_gondolas}_p-{num_part_pg}_size-{num_to_generate}_seed-{seed}.csv"
    file_path = os.path.join(dataset_path, file_name)

    if try_pregen and os.path.isfile(file_path):
        return pd.read_csv(file_path, index_col=0), df_health

    # else generate it
    # init ordering array
    random_order = np.arange(len(df_health))
    # initialize dataset storage
    dataset_storage = []

    # produce a number of elements
    for _ in tqdm(range(num_to_generate)):
        # generate sample element
        np.random.shuffle(random_order)
        sample = random_order[:num_gondolas*num_part_pg].reshape(num_gondolas, num_part_pg)

        # calc label
        label = build_wheel_happyness(df_health, sample)

        # add to storage
        dataset_storage.append(sample.flatten().tolist() + [label])

    # fuze dataset
    df_generated = pd.DataFrame(
        dataset_storage,
        columns=[
            f"g-{i}_p-{j}"
            for i in range(num_gondolas)
            for j in range(num_part_pg)
        ] + ['label']
    )

    # save dataset
    df_generated.to_csv(file_path)

    return df_generated, df_health
