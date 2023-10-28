import os.path

import numpy as np
import pandas as pd
import torch
from etnn.data import DEFAULT_DATA_PATH
from etnn.data.prepare_ferris_wheel import generate_ferris_dataset, add_valid_permutations, add_invalid_permutations


class FerrisWheelDataset(torch.utils.data.Dataset):
    def __init__(self, df_health, df_index):
        self.df_health = df_health
        self.df_index = df_index

        self.df_health.set_index('id', inplace=True)

    def __len__(self):
        return len(self.df_index)

    def __getitem__(self, idx):
        # Get the ID from the id_frame
        p_ids = self.df_index.iloc[idx, :-1]

        # Use the ID to index into the data_frame
        data = self.df_health.loc[p_ids]

        # Get the label from the last column of id_frame
        label = self.df_index.iloc[idx, -1]

        return torch.tensor(data.to_numpy(float), dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def load_pure_ferris_wheel_dataset(
        num_gondolas: int = 10,
        num_part_pg: int = 5,
        num_to_generate: int = 1_000,
        df_name_input: str = "Sleep_health_and_lifestyle_dataset.csv",
        dataset_path: str = DEFAULT_DATA_PATH,
        df_intermediate_output_name: str = 'health_dataset_preprocessed-1.csv',
        try_pregen: bool = True,
        seed: int = 4651431
):
    # load the datasets
    df_index, df_health = generate_ferris_dataset(
        num_gondolas=num_gondolas,
        num_part_pg=num_part_pg,
        num_to_generate=num_to_generate,
        dataset_path=dataset_path,
        df_intermediate_output_name=df_intermediate_output_name,
        df_name_input=df_name_input,
        try_pregen=try_pregen,
        seed=seed
    )

    return FerrisWheelDataset(df_health, df_index)


def load_modified_ferris_wheel_dataset(
        num_gondolas: int = 10,
        num_part_pg: int = 5,
        num_to_generate: int = 1_000,
        num_valid_to_add: int = 1_000,
        num_invalid_to_add: int = 1_000,
        df_name_input: str = "Sleep_health_and_lifestyle_dataset.csv",
        dataset_path: str = DEFAULT_DATA_PATH,
        df_intermediate_output_name: str = 'health_dataset_preprocessed-1.csv',
        try_pregen: bool = True,
        seed: int = 4651431
):
    # if pregenerated and parameter true, load dataset
    file_name = f"ferris-wheel_g-{num_gondolas}_p-{num_part_pg}_size-{num_to_generate}_seed-{seed}_valid" \
                f"-{num_valid_to_add}_invalid-{num_invalid_to_add}.csv"
    file_path = os.path.join(dataset_path, file_name)
    if try_pregen and os.path.isfile(file_path):
        return pd.read_csv(file_path)

    # seed randomness
    if seed is not None:
        np.random.seed(seed)

    # load the datasets
    df_index, df_health = generate_ferris_dataset(
        num_gondolas=num_gondolas,
        num_part_pg=num_part_pg,
        num_to_generate=num_to_generate,
        dataset_path=dataset_path,
        df_intermediate_output_name=df_intermediate_output_name,
        df_name_input=df_name_input,
        try_pregen=try_pregen,
        seed=seed
    )

    result_of_randomly_hitting_keyboard = 656658998

    # add valid permutations
    df_index = add_valid_permutations(
        num_add_equal_elem=num_valid_to_add,
        df_index=df_index,
        num_gondolas=num_gondolas,
        seed=np.random.randint(0, result_of_randomly_hitting_keyboard)
    )

    # add invalid permutations (invalid meaning valid permutations but with different labels - hence confusing the
    # equivariant system and the normal nn)
    df_index = add_invalid_permutations(
        num_add_nequal_elem=num_invalid_to_add,
        df_index=df_index,
        num_gondolas=num_gondolas,
        seed=np.random.randint(0, result_of_randomly_hitting_keyboard)
    )

    # save dataset
    df_index.to_csv(file_path)

    return FerrisWheelDataset(df_health, df_index)
