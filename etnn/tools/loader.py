import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler, Subset


def create_sampler(
        df_index: pd.DataFrame,
        dataset: Subset = None
) -> WeightedRandomSampler:
    # change name to working df
    working_df = df_index.copy()

    # if a dataset is provided - use indexes
    if dataset is not None:
        working_df = working_df.iloc[dataset.indices]

    # create a table for rounded labels
    working_df['rounded_label'] = working_df.label.round()

    # get counts for each label
    label_weights = len(working_df) / working_df.rounded_label.value_counts()

    # map the labels to the label weights
    working_df['label_weights'] = working_df.rounded_label.map(lambda x: label_weights.loc[x])

    return WeightedRandomSampler(
        working_df.label_weights,
        len(working_df),
        replacement=True
    )
