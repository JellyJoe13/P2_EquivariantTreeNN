import pandas as pd
import numpy as np
import typing

from etnn import TreeNode

"""
Rule ideas:
- People being happy with other people in same gondola
    + Age composition too seperated is bad
    + shift in gender is bad if too much, 50-50 is good or all one gender
    + same with age composition
    + sleep derived(multiplier with quality) persons get a subtraction and 'good sleepers' get bonus (sleep disorder 
      counts as stronger subtraction)
    + higher heart rate and pressure = joy or fear
    + composition of persons in regards to bmi : extreme values make others (no exception for group all those as to many
      underweight or overweight persons may be awquard as well)
- People being happy with neighboring gondolas composition
    + same age gets bonus, none gets penalty as potentially group is separated
    + gap between happyness index between self and neighbors causes it to produce a mean of only the neighbors
"""


def age_composition_score(
        df_elements: pd.DataFrame,
        id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
) -> float:
    """
    Function to calculate an age score of the group specified in the id list.

    :param df_elements: Dataframe containing the health data of the persons provided in the id list
    :type df_elements: pd.DataFrame
    :param id_list: list of ids of the persons in this group (gondola)
    :type id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
    :return: age score which is in [0, 1]
    :rtype: float
    """
    df_subset = df_elements[df_elements.id.isin(id_list)]

    score = 0

    age = df_subset.age.to_numpy()
    sorted_age = np.sort(age)

    # check if in 10 year gap
    gap = sorted_age[-1]-sorted_age[0]
    # print(gap)
    if gap <= 10:
        score = 1
        return score
    else:
        x = gap - 10
        score = (-1/60)*x + 1
        return score


def gender_composition_score(
        df_elements: pd.DataFrame,
        id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
) -> float:
    """
    Function to calculate a gender score of the group specified in the id list.

    :param df_elements: Dataframe containing the health data of the persons provided in the id list
    :type df_elements: pd.DataFrame
    :param id_list: list of ids of the persons in this group (gondola)
    :type id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
    :return: score which is in [0, 1]
    :rtype: float
    """
    df_sub = df_elements[df_elements.id.isin(id_list)]
    # other genders are considered neutral
    male = sum(df_sub.gender_male)
    female = sum(df_sub.gender_female)
    gsum = male + female

    return 2*min(male/gsum, female/gsum) if (male and female) else 1


def sleep_composition_score(
        df_elements: pd.DataFrame,
        id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
) -> float:
    """
    Function to calculate a sleep score of the group specified in the id list.

    :param df_elements: Dataframe containing the health data of the persons provided in the id list
    :type df_elements: pd.DataFrame
    :param id_list: list of ids of the persons in this group (gondola)
    :type id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
    :return: score which is in [0, 1]
    :rtype: float
    """
    df_sub = df_elements[df_elements.id.isin(id_list)]

    sleep_d_mean = df_sub.sleep_duration.mean()
    sleep_q_mean = df_sub.sleep_quality.mean()

    return min((sleep_q_mean/10)*(sleep_d_mean/7.5), 1)


def bmi_composition_score(
        df_elements: pd.DataFrame,
        id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
) -> float:
    """
    Function to calculate a bmi score of the group specified in the id list.

    :param df_elements: Dataframe containing the health data of the persons provided in the id list
    :type df_elements: pd.DataFrame
    :param id_list: list of ids of the persons in this group (gondola)
    :type id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
    :return: score which is in [0, 1]
    :rtype: float
    """
    df_sub = df_elements[df_elements.id.isin(id_list)]

    bmi = df_sub.bmi.to_numpy()
    bmi = np.where(bmi<2, 0, (bmi-1)/2)
    score = 1-np.sum(bmi)/len(df_sub)

    return score


def fear_composition_score(
        df_elements: pd.DataFrame,
        id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
) -> float:
    """
    Function to calculate a fear/heartrate score of the group specified in the id list.

    :param df_elements: Dataframe containing the health data of the persons provided in the id list
    :type df_elements: pd.DataFrame
    :param id_list: list of ids of the persons in this group (gondola)
    :type id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
    :return: score which is in [0, 1]
    :rtype: float
    """
    df_sub = df_elements[df_elements.id.isin(id_list)]

    blood_value = (df_sub.blood_pressure1 * df_sub.blood_pressure2 * df_sub.heart_rate).max()
    # print(blood_value)
    border = 900000

    if blood_value >= border:
        return 0
    else:
        return 1


def build_gondola_score(
        df_elements: pd.DataFrame,
        id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
) -> float:
    """
    Function to calculate a score of the group specified in the id list. This score is a weighted sum of the age,
    gender, sleep, bmi and fear scores introduced beforehand.

    :param df_elements: Dataframe containing the health data of the persons provided in the id list
    :type df_elements: pd.DataFrame
    :param id_list: list of ids of the persons in this group (gondola)
    :type id_list: typing.Union[typing.Iterable[int], typing.Iterable[float]]
    :return: score which is in [0, 10]
    :rtype: float
    """
    age_score = age_composition_score(df_elements, id_list)
    gender_score = gender_composition_score(df_elements, id_list)
    sleep_score = sleep_composition_score(df_elements, id_list)
    bmi_score = bmi_composition_score(df_elements, id_list)
    fear_score = fear_composition_score(df_elements, id_list)

    return 2.5*age_score + 2*gender_score + 0.5*sleep_score + 2*bmi_score + 3*fear_score


def build_wheel_happyness(
        df_elements: pd.DataFrame,
        wheel: np.ndarray[int]
) -> float:
    """
    Function that calculates the overall happyness of the ferris wheel which is to become a label to be predicted. Uses
    gondola wise scoring using function ``build_gondola_score(...)`` of own gondola, neighboring gondolas and age
    statistics of the neighboring gondolas to derive a new gondola happyness score which is summed up.

    :param df_elements: dataframe containing the person health data
    :type df_elements: pd.DataFrame
    :param wheel: numpy array containing the ids of the persons in the gondolas. Has shape
        ``(num_gondolas, num_persons_per_gondola)``
    :return: score of the ferris wheel, [0, num_gondolas * 10]
    :rtype: float
    """

    individual_scores = [
        build_gondola_score(df_elements, wheel[i])
        for i in range(len(wheel))
    ]
    neighbor_scores = [
        (individual_scores[(i-1)%len(wheel)] + individual_scores[(i+1)%len(wheel)]) / 2
        for i in range(len(wheel))
    ]
    age_mean = [
        df_elements[df_elements.isin(wheel[i])].age.mean()
        for i in range(len(wheel))
    ]
    age_group_bonus = [
        abs(age_mean[(i-1)%len(wheel)] - age_mean[i]) < 5 or abs(age_mean[(i+1)%len(wheel)] - age_mean[i]) < 5
        for i in range(len(wheel))
    ]

    return sum([
        min(
            (own + foreign) / 2 + 2*bonus,
            10
        )
        for own, foreign, bonus in zip(individual_scores, neighbor_scores, age_group_bonus)
    ])


def _reformat(result):
    if len(result.shape) != 2:
        result = result.reshape(1, -1)
    return result


def build_generative_label(
        tree: TreeNode,
        df_health: pd.DataFrame,
        map_element
):
    # Parameters
    k = 3  # equal to one left one right (or next 2 right/left elements)
    # generate embedding parameters
    if tree.node_type == "S":
        embedding_params = (np.arange(k) + 1) / (k*(k+1)/2)
    else:
        embedding_params = np.array([0.1, 0.8, 0.1])

    # assert that index is id column
    if 'id' in df_health:
        df = df_health.set_index('id', inplace=False)
    else:
        df = df_health

    if tree.node_type == "E":
        return _reformat(df.loc[map_element].to_numpy(float))

    elif tree.node_type == "P":
        sub_elem = []
        offset = 0
        for child in tree.children:
            sub_elem += [build_generative_label(
                tree=child,
                df_health=df,
                map_element=map_element[offset:offset+child.num_elem]
            )]
            offset += child.num_elem

        sub_elem = np.concatenate(sub_elem, axis=0)

        return _reformat(sub_elem.sum(axis=0))

    else:
        # get subelements
        sub_elem = []
        offset = 0
        for child in tree.children:
            sub_elem += [build_generative_label(
                tree=child,
                df_health=df,
                map_element=map_element[offset:offset+child.num_elem]
            )]
            offset += child.num_elem

        sub_elem = np.concatenate(sub_elem, axis=0)

        if tree.node_type == "S" or tree.node_type == "Q":  # ignore difference between s and q and treat it as q
            # shift stack
            shifted_emb = np.stack(
                [
                    sub_elem[2:],
                    sub_elem[1:-1],
                    sub_elem[:-2]
                ]
            )

        elif tree.node_type == "C":
            # shift stack
            # replace with previous times 0.1 and future times 0.1 and 0.8 itself
            shifted_emb = np.stack(
                [
                    np.concatenate([sub_elem[1:], sub_elem[0].reshape(1, -1)]),
                    sub_elem,
                    np.concatenate([sub_elem[-1].reshape(1, -1), sub_elem[:-1]])
                ]
            )

        # apply embedding parameters and sum up
        # print(shifted_emb.shape, embedding_params.shape)
        intermediate = np.einsum("abc,a->bc", shifted_emb, embedding_params)

        # simulate final linear layer
        final_embedding = ((np.arange(intermediate.shape[1]) + 1) / (intermediate.shape[1] * (intermediate.shape[1]+1)/2))[::-1]

        return _reformat(np.einsum("ab,b->b", intermediate, final_embedding))


def build_label_tree(
        df_health: pd.DataFrame,
        num_gondolas: int,
        num_part_pg: int,
        map_element: typing.Iterable[int],
        final_label_factor: int = 1/1000,
        mode: int = 0
) -> np.ndarray:
    # build the tree structure
    if mode == 0:
        tree = TreeNode("S", [
            TreeNode("P", [TreeNode("E", num_part_pg)])
            for _ in range(num_gondolas)
        ])
    elif mode == 1:
        tree = TreeNode("C", [
            TreeNode("P", [TreeNode("E", num_part_pg)])
            for _ in range(num_gondolas)
        ])
    else:
        raise Exception("not valid option chosen")

    # calculate the label (intermediate)
    label = build_generative_label(
        tree=tree,
        df_health=df_health,
        map_element=map_element
    )

    return label.sum() * final_label_factor
