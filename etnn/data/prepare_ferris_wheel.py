import pandas as pd
import numpy as np
import os

from etnn import TreeNode
from etnn.data import DEFAULT_DATA_PATH
from tqdm import tqdm
from etnn.data.ferris_score_helpers import build_wheel_happyness, build_generative_label, build_label_tree
import typing
from multiprocess.pool import Pool


def prepare_1_ferris(
        df_name_input: str = "Sleep_health_and_lifestyle_dataset.csv",
        dataset_path: str = DEFAULT_DATA_PATH,
        df_name_output: str = 'health_dataset_preprocessed-1.csv',
        try_pregen: bool = True
) -> pd.DataFrame:
    """
    Function that preprocesses the person health dataset once in respect to creating the ferris wheel dataset.

    :param df_name_input: name of the dataframe to process, default: ``Sleep_health_and_lifestyle_dataset.csv``.
    :type df_name_input: str
    :param dataset_path: Path to the dataset folder, default: ``DEFAULT_DATA_PATH``
    :type dataset_path: str
    :param df_name_output: name of the file into which to save the preprocessed dataset, default:
        ``health_dataset_preprocessed-1.csv``.
    :type df_name_output: str
    :param try_pregen: determines whether a pre-existing preprocessed data file should be used or not, default: ``True``
    :type try_pregen: bool
    :return: dataframe containing the preprocessed data
    :rtype: pd.DataFrame
    """
    # CHECK IF PRECOMPUTED
    if try_pregen and os.path.isfile(os.path.join(dataset_path, df_name_output)):
        return pd.read_csv(os.path.join(dataset_path, df_name_output))

    # LOADING
    # join path
    df_path = os.path.join(dataset_path, df_name_input)

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


def normalize_dataset(
        df_health: pd.DataFrame
) -> pd.DataFrame:
    """
    Normalize the provided dataset with sklearn min max scaler. Excludes a possible ``'id'`` column.

    :param df_health: dataframe to be normaized.
    :type df_health: pd.DataFrame
    :return: normalized dataframe
    :rtype: pd.DataFrame
    """
    cols = df_health.columns
    cols = [
        x
        for x in cols
        if 'id' != x
    ]

    from sklearn import preprocessing

    scaler = preprocessing.MinMaxScaler()

    df_out = df_health.copy()

    df_out[cols] = scaler.fit_transform(df_health[cols].values)

    return df_out


def generate_ferris_dataset(
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
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function that generates the ferris dataset's core dataset, the health dataset and the index dataset.

    :param num_gondolas: number of gondolas the ferris wheel should have, default: ``10``
    :type num_gondolas: int
    :param num_part_pg: number of people in each gondola, default: ``5``
    :type num_part_pg: int
    :param num_to_generate: number of elements the core dataset should have, default: ``1000``
    :type num_to_generate:  int
    :param df_name_input: name of the dataset input, default: ``Sleep_health_and_lifestyle_dataset.csv``
    :type df_name_input: str
    :param dataset_path: path to dataset folder, default: ``DEFAULT_DATA_PATH``
    :type dataset_path: str
    :param df_intermediate_output_name: name of intermediate dataset csv, default: ``health_dataset_preprocessed-1.csv``
    :type df_intermediate_output_name: str
    :param try_pregen: determines whether it shall be attempted to load the pre-generated csv files, default: ``True``
    :type try_pregen: bool
    :param seed: seed to be used for random generation procedures
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
    :return: health dataset and index dataset
    :rtype: typing.Tuple[pd.DataFrame, pd.DataFrame]
    """
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

    # if normalize - normalize dataset
    if normalize:
        df_health = normalize_dataset(df_health)

    # see if file already exists and load this
    # file name logic
    file_name = f"ferris-wheel_g-{num_gondolas}_p-{num_part_pg}_size-{num_to_generate}_seed-{seed}_label-{label_type}"
    if label_type == "tree":
        file_name += f"-{final_label_factor}"
    if normalize:
        file_name += f"-normalized"
    file_name += f".csv"
    file_path = os.path.join(dataset_path, file_name)

    if try_pregen and os.path.isfile(file_path):
        return pd.read_csv(file_path, index_col=0), df_health

    # else generate it
    # init ordering array
    random_order = np.arange(len(df_health))+1

    # initialize dataset storage
    dataset_storage = []
    data_store = []

    # produce a number of elements
    for _ in tqdm(range(num_to_generate)):
        # generate sample element
        np.random.shuffle(random_order)
        sample = random_order[:num_gondolas*num_part_pg]
        if label_type == "default":
            sample = sample.reshape(num_gondolas, num_part_pg)

        data_store += [sample.copy()]

    # calc label
    if label_type == "tree":
        func = lambda x: x.flatten().tolist() + [build_label_tree(
            df_health=df_health,
            map_element=x,
            num_gondolas=num_gondolas,
            num_part_pg=num_part_pg,
            final_label_factor=final_label_factor*1000 if normalize else final_label_factor,
            mode=0
        )]

    elif label_type == "tree_advanced":
        func = lambda x: x.flatten().tolist() + [build_label_tree(
            df_health=df_health,
            map_element=x,
            num_gondolas=num_gondolas,
            num_part_pg=num_part_pg,
            final_label_factor=final_label_factor*1000 if normalize else final_label_factor,
            mode=1
        )]

    else:
        func = lambda x: x.flatten().tolist() + [build_wheel_happyness(df_health, x)]

    with Pool(processes=os.cpu_count()) as p:
        dataset_storage = list(p.imap(func, tqdm(data_store)))

    # dataset_storage = list(map(func, tqdm(data_store)))

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


def generate_ferris_dataset_single_node(
        node_type,
        num_elem,
        num_to_generate,
        dataset_path,
        seed,
        final_label_factor,
        normalize,
        df_name_input: str = "Sleep_health_and_lifestyle_dataset.csv",
        df_intermediate_output_name: str = 'health_dataset_preprocessed-1.csv'
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
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
    # set seed if not none
    if seed is not None:
        np.random.seed(seed)

    # load dataset which is the base of the data generation
    df_health = prepare_1_ferris(
        df_name_input=df_name_input,
        dataset_path=dataset_path,
        df_name_output=df_intermediate_output_name,
        try_pregen=True
    )

    # if normalize - normalize dataset
    if normalize:
        df_health = normalize_dataset(df_health)

    # see if file already exists and load this
    # file name logic
    file_name = f"ferris-wheel_e-{num_elem}-type-{node_type}_size-{num_to_generate}_seed-{seed}_{final_label_factor}"
    if normalize:
        file_name += f"-normalized"
    file_name += f".csv"
    file_path = os.path.join(dataset_path, file_name)

    if os.path.isfile(file_path):
        return pd.read_csv(file_path, index_col=0), df_health

    # else generate it
    # init ordering array
    random_order = np.arange(len(df_health))+1

    # initialize dataset storage
    dataset_storage = []
    data_store = []

    # produce a number of elements
    for _ in tqdm(range(num_to_generate)):
        # generate sample element
        np.random.shuffle(random_order)
        sample = random_order[:num_elem]

        data_store += [sample.copy()]

    # calc label
    factor = final_label_factor * 1000 if normalize else final_label_factor
    func = lambda x: x.flatten().tolist() + [build_generative_label(
        tree=TreeNode(node_type, [TreeNode("E", num_elem)]),
        df_health=df_health,
        map_element=x
    ).sum() * factor]

    with Pool(processes=os.cpu_count()) as p:
        dataset_storage = list(p.imap(func, tqdm(data_store)))

    # dataset_storage = list(map(func, tqdm(data_store)))
    #dataset_storage = [
    #    build_generative_label(
    #        tree=TreeNode(node_type, [TreeNode("E", num_elem)]),
    #        df_health=df_health,
    #        map_element=x
    #    ).sum()
    #    for x in data_store
    #]

    # fuze dataset
    df_generated = pd.DataFrame(
        dataset_storage,
        columns=[
                    f"e-{i}"
                    for i in range(num_elem)
                ] + ['label']
    )

    # save dataset
    df_generated.to_csv(file_path)

    return df_generated, df_health


def add_valid_permutations(
        num_add_equal_elem: int,
        df_index: pd.DataFrame,
        num_gondolas: int,
        seed: int = 4354353
) -> pd.DataFrame:
    """
    Function that permutes some sampled elements in the index dataframe and adds them with the correct value to the
    dataframe that is returned.

    :param num_add_equal_elem: number of equivariant elements to add
    :type num_add_equal_elem: int
    :param df_index: dataframe from which to sample elements and which to enlarge with this operations newly permutated
        elements
    :type df_index: pd.DataFrame
    :param num_gondolas: number of gondolas the ferris wheel has
    :type num_gondolas: int
    :param seed: seed to use for random permutation operations
    :type seed: int
    :return: the enlarged df_index dataset with correctly labeled elements
    :rtype: pd.DataFrame
    """
    if seed is not None:
        np.random.seed(seed)

    merged = df_index

    remaining_to_generate = num_add_equal_elem

    while True:
        pd_new = sample_new_permutations(
            df_index=merged,
            num_elem=remaining_to_generate,
            num_gondolas=num_gondolas,
            seed=np.random.randint(0, 2 ** 30)
        )

        # concatenate and return
        merged = df_index.merge(pd_new, how='outer', indicator=True)

        remaining_to_generate = sum(merged['_merge'] == 'both')
        merged = merged.drop('_merge', axis=1)

        if remaining_to_generate == 0:
            break

    return pd.concat([df_index, pd_new], ignore_index=True)


def sample_new_permutations(
        df_index: pd.DataFrame,
        num_elem: int,
        num_gondolas: int,
        seed: int = None,
        merge_check: bool = False
) -> pd.DataFrame:
    """
    Function to create permutations to the ferris wheel dataset and returning it as a pandas DataFrame.

    :param df_index: dataframe which elements to sample and randomly permute
    :type df_index: pd.DataFrame
    :param num_elem: number of elements to create
    :type num_elem: int
    :param num_gondolas: number of gondolas the dataset has. can be also read from the dataframe by the column names
        but not used in this method
    :type num_gondolas: int
    :param seed: seed to use to control randomness, default: ``None``
    :type seed: int
    :param merge_check: boolean parameters controlling whether a merge check should be performed. A merge check contains
        whether it should be checked if the random permutation produced an element which is already in df_index
    :type merge_check: bool
    :return: pandas dataframe containing the newly created elements
    :rtype: pd.DataFrame
    """
    # create viable permutations of input and add them to dataset
    # sample elements to perturb
    df_sampled = df_index.sample(num_elem, replace=True, random_state=seed)
    if seed is not None:
        np.random.seed(seed)

    # for each randomly one of two things:
    # - group order change
    #   + shift groups
    #   + invert group order (left out for now as easier this way)
    # - permutate gondola people

    # convert to numpy
    df_t = df_sampled.to_numpy(int)[:, :-1]

    # changing shape to make dimensions match ferris wheel structure
    df_g = df_t.reshape(df_t.shape[0], num_gondolas, -1)

    # create shifts and index
    shifts = np.random.randint(0, num_gondolas, df_g.shape[0])
    idx = np.arange(num_gondolas)

    # perturbing elements in numpy array in manner of C and P
    for i in tqdm(range(num_elem)):
        df_g[i] = df_g[i][(idx + shifts[i]) % num_gondolas]
        for j in range(num_gondolas):
            np.random.shuffle(df_g[i][j])

    # create dataset out of perturbed elements
    pd_new = pd.DataFrame(
        df_t,
        columns=df_sampled.columns[:-1]
    )
    pd_new[df_sampled.columns[-1]] = df_sampled.label.to_numpy()

    if merge_check:
        merge_checked = pd_new.merge(df_index, how='outer', indicator=True)
        num_duplicate = sum(merge_checked['_merge'] == 'both')

        good_samples = merge_checked[merge_checked['_merge'] == 'left_only'].copy()
        good_samples = good_samples.drop('_merge', axis=1)
        merge_checked = merge_checked.drop('_merge', axis=1)

        if num_duplicate == 0:
            return good_samples

        # get replacements for the duplicates
        new_samples = sample_new_permutations(
            df_index=merge_checked,
            num_elem=num_duplicate,
            num_gondolas=num_gondolas,
            seed=np.random.randint(0, 2 ** 30),
            merge_check=True
        )

        return good_samples.merge(new_samples, how='outer')

    return pd_new


def add_invalid_permutations(
        num_add_nequal_elem: int,
        df_index: pd.DataFrame,
        num_gondolas: int,
        seed: int = 4354353
) -> pd.DataFrame:
    """
    Function that permutes some sampled elements in the index dataframe and adds them with the perturbed value to the
    dataframe that is returned.

    :param num_add_nequal_elem: number of equivariant elements to add (with perturbed label)
    :type num_add_nequal_elem: int
    :param df_index: dataframe from which to sample elements and which to enlarge with this operations newly permutated
        elements
    :type df_index: pd.DataFrame
    :param num_gondolas: number of gondolas the ferris wheel has
    :type num_gondolas: int
    :param seed: seed to use for random permutation operations
    :type seed: int
    :return: the enlarged df_index dataset with falsely labeled elements
    :rtype: pd.DataFrame
    """
    # create viable permutations of input and add them to dataset
    # sample elements to perturb
    df_sampled = df_index.sample(num_add_nequal_elem, replace=True, random_state=seed)
    if seed is not None:
        np.random.seed(seed)

    # for each randomly one of two things:
    # - group order change
    #   + shift groups
    #   + invert group order (left out for now as easier this way)
    # - permutate gondola people

    # convert to numpy
    df_t = df_sampled.to_numpy()[:, :-1]

    # changing shape to make dimensions match ferris wheel structure
    df_g = df_t.reshape(df_t.shape[0], num_gondolas, -1)

    # create shifts and index
    shifts = np.random.randint(0, num_gondolas, df_g.shape[0])
    idx = np.arange(num_gondolas)

    # perturbing elements in numpy array in manner of C and P
    for i in tqdm(range(num_add_nequal_elem)):
        df_g[i] = df_g[i][(idx + shifts[i]) % num_gondolas]
        for j in range(num_gondolas):
            np.random.shuffle(df_g[i][j])

    # create dataset out of perturbed elements
    pd_new = pd.DataFrame(
        np.c_[
            df_t,
            np.random.rand(len(df_t))*20
        ],
        columns=df_sampled.columns
    )

    # concatenate and return
    return pd.concat([df_index, pd_new], ignore_index=True)
