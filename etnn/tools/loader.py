import pandas as pd
from torch.utils.data import WeightedRandomSampler, Subset


def create_sampler(
        df_index: pd.DataFrame,
        dataset: Subset = None
) -> WeightedRandomSampler:
    """
    Create a pytorch sampler that smooths uneven distribution.

    :param df_index: dataset for which to create the sampler for
    :type df_index: pd.DataFrame
    :param dataset: dataset of the split original dataset if applicable - else ``None``. This provides the information
        which element (by index) belongs to the current dataset and in which order the elements are contained in the
        subset.
    :type dataset: Subset
    :return: Random Sampler that randomly samples elements with repetition
    :rtype: WeightedRandomSampler
    """
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
