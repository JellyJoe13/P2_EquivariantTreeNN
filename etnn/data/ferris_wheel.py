import os.path
import typing

import numpy as np
import pandas as pd
import torch
from etnn.data import DEFAULT_DATA_PATH
from etnn.data.prepare_ferris_wheel import generate_ferris_dataset, add_valid_permutations, add_invalid_permutations, \
    prepare_1_ferris, generate_ferris_dataset_single_node, normalize_dataset


class FerrisWheelDataset(torch.utils.data.Dataset):
    """
    Pytorch dataset of the ferris wheel dataset concept. Each element in the dataset consists of data x and a label y.

    y in this case is the happyness score calculated by the function ``build_wheel_happyness(...)``.

    x consists of gondolas which contains a certain number of people, which is specified by the input dataset. The logic
    how many gondolas and persons per gondolas are is not explicitly contained in this object but contained in the
    element ``df_index`` through column names. The tensor provided by this function however will not be in any
    specific order or structure that indicates which data is assigned to which gondola.
    """
    def __init__(self, df_health, df_index, num_gondolas):
        """
        Init function of the pytorch FerrisWheelDataset, which inherits the pytorch Dataset object.

        :param df_health: Person health dataset
        :type df_health: pd.DataFrame
        :param df_index: Gondola dataset containing person ids and happyness score (=label)
        :type df_index: pd.DataFrame
        :param num_gondolas: Number of gondolas in ferris wheel
        :type num_gondolas: int
        """
        self.df_health = df_health.set_index('id', inplace=False) if 'id' in df_health else df_health
        self.df_index = df_index

        self.num_gondolas = num_gondolas

    def __len__(self):
        """
        Generic length function.

        :return: The number of elements contained in the dataset.
        :rtype: int
        """
        return len(self.df_index)

    def __getitem__(self, idx):
        """
        Returns the item stored in the dataset at position idx.

        :param idx: index of the item to return
        :type idx: int
        :return: Data and label of this index. Data has shape
            ``(num_gondolas*num_persons_in_gondola, person_data_dim)``, label is a simple 1d tensor containing one
            float value.
        :rtype: typing.Tuple[torch.Tensor, torch.Tensor]
        """
        # Get the ID from the id_frame
        p_ids = self.df_index.iloc[idx, :-1]

        # Use the ID to index into the data_frame
        data = self.df_health.loc[p_ids]

        # Get the label from the last column of id_frame
        label = self.df_index.iloc[idx, -1]

        return torch.tensor(data.to_numpy(float), dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def to_classification(
            self,
            num_classes: int
    ):
        """
        Converts the regression type dataset into a classification type dataset by dividing the continuous labels
        through their maximum amount and and multiplying them with the desired number of classes - then round.

        :param num_classes: number of classes for the new classification dataset
        :type num_classes: int
        :return: Nothing
        """
        self.df_index.label = (self.df_index.label/(10*self.num_gondolas)*num_classes).round()

    def post_normalize(self):
        """
        Function that normalizes the body of the dataset, namely the stored person health data. Does not affect labels.

        :return: Nothing
        """
        self.df_health = normalize_dataset(self.df_health)


def load_pure_ferris_wheel_dataset(
        num_gondolas: int = 10,
        num_part_pg: int = 5,
        num_to_generate: int = 1_000,
        df_name_input: str = "Sleep_health_and_lifestyle_dataset.csv",
        dataset_path: str = DEFAULT_DATA_PATH,
        df_intermediate_output_name: str = 'health_dataset_preprocessed-1.csv',
        try_pregen: bool = True,
        seed: int = 4651431,
        label_type: str = "default",
        final_label_factor: float = 1/1000,
        normalize: bool = False
) -> typing.Tuple[FerrisWheelDataset, pd.DataFrame]:
    """
    Function that loads dataset from pre-generated csv files or generates data(and dataset) (and saves these
    intermediate csv files). Loads a dataset that exactly follows the allowed permutations combined with the exact
    calculated happyness scores(or other scores).

    :param num_gondolas: Number of gondolas the ferris wheel should have, default: ``15``
    :type num_gondolas: int
    :param num_part_pg: Number of persons in each gondola, default: ``5``. (by default no empty passengers/seats will be
        generated)
    :type num_part_pg: int
    :param num_to_generate: Number of dataset entries to generate, default: ``1000``
    :type num_to_generate: int
    :param df_name_input: Name of the dataset to use as the input for the persons health data, default:
        ``Sleep_health_and_lifestyle_dataset.csv``
    :type df_name_input: str
    :param dataset_path: Specifies the path to the dataset folder, default ``DEFAULT_DATA_PATH``
    :type dataset_path: str
    :param df_intermediate_output_name: Name of the intermediate datasets output, default:
        ``'health_dataset_preprocessed-1.csv'``. Intermediate dataset denotes the preprocessed health dataset.
    :type df_intermediate_output_name: str
    :param try_pregen: Controls whether pre-generated shall be loaded or new data should be generated hence overwriting
        and updating previously generated data with the same parameters, default: ``True``
    :type try_pregen: bool
    :param seed: Seed to use for random generation
    :type seed: int
    :param label_type: label to be generated. Viable options: ``'default'`` producing the most complex label with logic
        inspired by a real ferris wheel, ``'tree'`` and ``'tree_advanced'`` for a tree based generation of label with
        minimal logic behind this label. Difference in first and second tree option is that the first option is a
        downgraded version only using S and P type labels and not C type labels instead of C as it would have been
        intended for a ferris wheel dataset, default: ``'default'``.
    :type label_type: str
    :param final_label_factor: Factor by which the label is scaled. Does not apply for ``'default'`` option in
        label_type parameter. Default: ``1/1000``
    :type final_label_factor: float
    :param normalize: Whether or not the data is normalized with a min-max scaler, default: ``False``
    :type normalize: bool
    :return: generated dataset and index dataframe for data loading purposes
    :rtype: typing.Tuple[FerrisWheelDataset, pd.DataFrame]
    """
    # load the datasets
    df_index, df_health = generate_ferris_dataset(
        num_gondolas=num_gondolas,
        num_part_pg=num_part_pg,
        num_to_generate=num_to_generate,
        dataset_path=dataset_path,
        df_intermediate_output_name=df_intermediate_output_name,
        df_name_input=df_name_input,
        try_pregen=try_pregen,
        seed=seed,
        label_type=label_type,
        final_label_factor=final_label_factor,
        normalize=normalize
    )

    return FerrisWheelDataset(df_health, df_index, num_gondolas), df_index


def load_pure_ferris_wheel_dataset_single_node(
        node_type: str,
        num_elem: int,
        num_to_generate: int,
        dataset_path: str,
        final_label_factor: float = 1/1000,
        normalize: bool = False,
        seed: int = 4651431,
        df_name_input: str = "Sleep_health_and_lifestyle_dataset.csv",
        df_intermediate_output_name: str = 'health_dataset_preprocessed-1.csv'
) -> typing.Tuple[torch.utils.data.Dataset, pd.DataFrame]:
    """
    Function that creates a dataset and data fame for generating data generated with the logic of a permutation tree
    with a singular node as its root with all elements.

    :param node_type: Type of the label of the singular permutation tree in its defining permutation tree. Determines
        how the label is generated.
    :type node_type: str
    :param num_elem: Number of elements to be grouped belonging to the singular permutation tree node.
    :type num_elem: int
    :param num_to_generate: number of elements to generate for the dataset.
    :type num_to_generate: int
    :param dataset_path: path to the dataset folder
    :type dataset_path: str
    :param final_label_factor: final label to be applied to the original label, default: ``1/1000``.
    :type final_label_factor: float
    :param normalize: bool controlling whether the input data with person health data should be normalized or not.
    :type normalize: bool
    :param seed: seed to make experiments reproducible, default: ``4651431``.
    :type seed: int
    :param df_name_input: name of the dataframe to load, default: ``'Sleep_health_and_lifestyle_dataset.csv'``.
    :type df_name_input: str
    :param df_intermediate_output_name: name of the intermediate (preprocessed) data to produce and store or load.
        Default: ``'health_dataset_preprocessed-1.csv'``.
    :type df_intermediate_output_name: str
    :return: dataset and data frame containing structural information
    :rtype: typing.Tuple[torch.utils.data.Dataset, pd.DataFrame]
    """
    # load the datasets
    df_index, df_health = generate_ferris_dataset_single_node(
        node_type=node_type,
        num_elem=num_elem,
        num_to_generate=num_to_generate,
        dataset_path=dataset_path,
        seed=seed,
        final_label_factor=final_label_factor,
        normalize=normalize,
        df_name_input=df_name_input,
        df_intermediate_output_name=df_intermediate_output_name
    )

    return FerrisWheelDataset(df_health, df_index, num_elem), df_index


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
        seed: int = 4651431,
        label_type: str = "default",
        final_label_factor: float = 1/1000
) -> typing.Tuple[FerrisWheelDataset, pd.DataFrame]:
    """
    Function that loads dataset from pre-generated csv files or generates data(and dataset) (and saves these
    intermediate csv files). Loads a dataset that not exactly follows the allowed permutations combined with the
    exact calculated happyness scores. It contains these items in the beginning and then adds permutated pure
    items to increase the potential benefit of equivariant nn's but afterwards adds label perturbed data.

    :param num_gondolas: Number of gondolas the ferris wheel should have, default: ``15``
    :type num_gondolas: int
    :param num_part_pg: Number of persons in each gondola, default: ``5``. (by default no empty passengers/seats will be
        generated)
    :type num_part_pg: int
    :param num_to_generate: Number of dataset entries to generate, default: ``1000``
    :type num_to_generate: int
    :param num_valid_to_add: Number of pure items to add which (permuted) already exist within the original dataset,
        default: ``1000``
    :type num_valid_to_add: int
    :param num_invalid_to_add: Number of perturbed items to add to the previously pure data to challenge the neural
        networks capability to react to falsified data, default: ``1000``
    :type num_invalid_to_add: int
    :param df_name_input: Name of the dataset to use as the input for the persons health data, default:
        ``Sleep_health_and_lifestyle_dataset.csv``
    :type df_name_input: str
    :param dataset_path: Specifies the path to the dataset folder, default ``DEFAULT_DATA_PATH``
    :type dataset_path: str
    :param df_intermediate_output_name: Name of the intermediate datasets output, default:
        ``'health_dataset_preprocessed-1.csv'``. Intermediate dataset denotes the preprocessed health dataset.
    :type df_intermediate_output_name: str
    :param try_pregen: Controls whether pre-generated shall be loaded or new data should be generated hence overwriting
        and updating previously generated data with the same parameters, default: ``True``
    :type try_pregen: bool
    :param seed: Seed to use for random generation
    :type seed: int
    :param label_type: label to be generated. Viable options: ``'default'`` producing the most complex label with logic
        inspired by a real ferris wheel, ``'tree'`` and ``'tree_advanced'`` for a tree based generation of label with
        minimal logic behind this label. Difference in first and second tree option is that the first option is a
        downgraded version only using S and P type labels and not C type labels instead of C as it would have been
        intended for a ferris wheel dataset, default: ``'default'``.
    :type label_type: str
    :param final_label_factor: Factor by which the label is scaled. Does not apply for ``'default'`` option in
        label_type parameter. Default: ``1/1000``
    :type final_label_factor: float
    :return: generated dataset and index dataframe for data loading purposes
    :rtype: typing.Tuple[FerrisWheelDataset, pd.DataFrame]
    """
    # if pregenerated and parameter true, load dataset
    file_name = f"ferris-wheel_g-{num_gondolas}_p-{num_part_pg}_size-{num_to_generate}_seed-{seed}_valid" \
                f"-{num_valid_to_add}_invalid-{num_invalid_to_add}_label-{label_type}"

    if label_type == "tree":
        file_name += f"-{final_label_factor}"

    file_name += f".csv"
    file_path = os.path.join(dataset_path, file_name)
    if try_pregen and os.path.isfile(file_path):
        df_health = prepare_1_ferris(
            df_name_input=df_name_input,
            dataset_path=dataset_path,
            df_name_output=df_intermediate_output_name,
            try_pregen=try_pregen
        )
        df_index = pd.read_csv(file_path, index_col=0)
        return FerrisWheelDataset(df_health, df_index, num_gondolas), df_index

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
        seed=seed,
        label_type=label_type,
        final_label_factor=final_label_factor
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

    return FerrisWheelDataset(df_health, df_index, num_gondolas), df_index


def prepare_pure_test_dataset(
        df_index: pd.DataFrame,
        train_indices: typing.Iterable[int],
        num_gondolas: int,
        num_to_generate: int = 1_000,
        df_name_input: str = "Sleep_health_and_lifestyle_dataset.csv",
        dataset_path: str = DEFAULT_DATA_PATH,
        df_intermediate_output_name: str = 'health_dataset_preprocessed-1.csv',
        try_pregen: bool = True,
        seed: int = 4651431
) -> typing.Tuple[FerrisWheelDataset, pd.DataFrame]:
    """
    Function that creates 'pure' dataset based on training set. Hence has already been learned in the ETNN.

    :param df_index: Dataframe that describes the ferris wheel guests with an id
    :type df_index: pd.DataFrame
    :param train_indices: Indices of the split dataset or sub-dataset corresponding to the training dataset
    :type train_indices: typing.Iterable[int]
    :param num_gondolas: Number of gondolas the ferris wheel should have, default: ``15``
    :type num_gondolas: int
    :param num_to_generate: Number of dataset entries to generate, default: ``1000``
    :type num_to_generate: int
    :param df_name_input: Name of the dataset to use as the input for the persons health data, default:
        ``'Sleep_health_and_lifestyle_dataset.csv'``
    :type df_name_input: str
    :param dataset_path: Specifies the path to the dataset folder, default ``DEFAULT_DATA_PATH``
    :type dataset_path: str
    :param df_intermediate_output_name: Name of the intermediate datasets output, default:
        ``'health_dataset_preprocessed-1.csv'``. Intermediate dataset denotes the preprocessed health dataset.
    :type df_intermediate_output_name: str
    :param try_pregen: Controls whether pre-generated shall be loaded or new data should be generated hence overwriting
        and updating previously generated data with the same parameters, default: ``True``
    :type try_pregen: bool
    :param seed: Seed to use for random generation
    :type seed: int
    :return: generated dataset and index dataframe for data loading purposes
    :rtype: typing.Tuple[FerrisWheelDataset, pd.DataFrame]
    """
    # get health dataset
    df_health = prepare_1_ferris(
        df_name_input=df_name_input,
        dataset_path=dataset_path,
        df_name_output=df_intermediate_output_name,
        try_pregen=try_pregen
    )

    # filter out training index if it differs
    if train_indices is not None:
        df_index_train = df_index[df_index.index.isin(train_indices)]
    else:
        df_index_train = df_index

    # generate valid(pure) permutations
    df_index_test = add_valid_permutations(
        num_add_equal_elem=num_to_generate,
        df_index=df_index_train,
        num_gondolas=num_gondolas,
        seed=seed
    )
    return FerrisWheelDataset(
        df_health,
        df_index_test,
        num_gondolas
    ), df_index_test
