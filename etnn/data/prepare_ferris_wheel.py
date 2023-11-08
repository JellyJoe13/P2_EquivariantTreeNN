import pandas as pd
import numpy as np
import os
from etnn.data import DEFAULT_DATA_PATH
from tqdm import tqdm
from etnn.data.ferris_score_helpers import build_wheel_happyness
import typing
from multiprocess.pool import Pool


def prepare_1_ferris(
        df_name_input: str = "Sleep_health_and_lifestyle_dataset.csv",
        dataset_path: str = DEFAULT_DATA_PATH,
        df_name_output: str = 'health_dataset_preprocessed-1.csv',
        try_pregen: bool = True
) -> pd.DataFrame:
    """
    Function that preprocesses the person health dataset once in repsect to creating the ferris wheel dataset.

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
    :param dataset_path: path to dataset folder, default: ``DEFAULT_DATA_PATH´´
    :type dataset_path: str
    :param df_intermediate_output_name: name of intermediate dataset csv, default: ``health_dataset_preprocessed-1.csv``
    :type df_intermediate_output_name: str
    :param try_pregen: determines whether it shall be attempted to load the pre-generated csv files, default: ``True``
    :type try_pregen: bool
    :param seed: seed to be used for random generation procedures
    :type seed: int
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

    # see if file already exists and load this
    # file name logic
    file_name = f"ferris-wheel_g-{num_gondolas}_p-{num_part_pg}_size-{num_to_generate}_seed-{seed}.csv"
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
        sample = random_order[:num_gondolas*num_part_pg].reshape(num_gondolas, num_part_pg)

        data_store += [sample.copy()]

    # calc label
    func = lambda x: x.flatten().tolist() + [build_wheel_happyness(df_health, x)]

    with Pool(processes=os.cpu_count()) as p:
        dataset_storage = list(p.imap(func, tqdm(data_store)))

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
