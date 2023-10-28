import torch
from etnn.data import DEFAULT_DATA_PATH
from etnn.data.prepare_ferris_wheel import generate_ferris_dataset


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
