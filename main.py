import numpy as np
import pandas as pd
from DataSet import DataSet

epsilon = np.finfo(float).eps


def find_entropy(df):
    """
    Funkcja zwracająca entropię zmiennej objaśnianej (target).
    """
    label = df.keys()[-1]
    entropy = 0
    values = df[label].unique()
    for value in values:
        fraction = df[label].value_counts()[value] / len(df[label])
        entropy += -fraction * np.log2(fraction)
    return entropy


def find_entropy_attribute(df, attribute):
    """
    Funkcja zwraca entropię dla wskazanego atrybutu.
    """
    label = df.keys()[-1]
    target_variables = df[label].unique()
    variables = df[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[label] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + epsilon)
            entropy += - fraction * np.log2(fraction + epsilon)
        fraction2 = den / len(df)
        entropy2 += - fraction2 * entropy
    return abs(entropy2)


def find_winner(df):
    """
    Funkcja zwraca nazwę atrybutu według, którego nastąpi podział (atrybut
    posiadający największy zysk informacyjny).
    """
    entropy_attr = []
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df) - find_entropy_attribute(df, key))
    return df.keys()[:-1][np.argmax(IG)]


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def build_tree(df, tree=None):
    label = df.keys()[-1]

    # atrybut z największą wartością zysku informacyjnego
    node = find_winner(df)

    # unikalne wartości dla wskazanego atrybutu
    att_values = np.unique(df[node])

    # utworzenie pustego słownika do przechowania drzewa
    if tree is None:
        tree = {}
        tree[node] = {}

    # print(node)
    # wykonujemy pętlę aby zbudować drzewo, jeśli podzbiór jest czysty
    # zatrzymujemy działanie pętli
    for value in att_values:

        subtable = get_subtable(df, node, value)
        class_value, counts = np.unique(subtable['item_class'], return_counts=True)
        # print(class_value)
        if len(counts) == 1:
            tree[node][value] = class_value[0]
        else:
            tree[node][value] = build_tree(subtable)

    return tree


def predict(inst, tree):
    for nodes in tree.keys():

        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0

        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break
    return prediction


ds = DataSet()
training_dataset = ds.generate_training_dataset()
test_dataset = ds.generate_test_dataset()
df_training = pd.DataFrame(training_dataset)
df_test = pd.DataFrame(test_dataset)
print(training_dataset)
tree = build_tree(df_training)
print(tree)

for index, image_meta_data in df_test.iterrows():
    # print(image_meta_data)
    print(predict(image_meta_data, tree))
