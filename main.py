import os
from datetime import datetime

import numpy as np
import pandas as pd
from DataSet import DataSet
import json
import logging

epsilon = np.finfo(float).eps

logger = logging.getLogger()
logger.setLevel(logging.INFO)

fh = logging.FileHandler('log\\K4_{:%Y-%m-%d_%H_%M_%S}.log'.format(datetime.now()))
fh.setLevel(logging.INFO)

logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())


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
    return abs(entropy2) if abs(entropy2) > 1.0e-3 else 0


def find_winner(df):
    """
    Funkcja zwraca nazwę atrybutu według, którego nastąpi podział (atrybut
    posiadający największy zysk informacyjny).
    """
    logger.info("Badany zbiór: {}".format(df.to_string()))
    entropy_attr = []
    IG = []
    logger.info("Entropia badanego zbioru: {}".format(find_entropy(df)))
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df) - find_entropy_attribute(df, key))

    logger.info("Entropia zwycięskiego atrybutu({}): {}".format(
        df.keys()[:-1][np.argmax(IG)],
        find_entropy_attribute(df, df.keys()[:-1][np.argmax(IG)]))
    )
    return df.keys()[:-1][np.argmax(IG)]


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def build_tree(df, tree=None):
    label = df.keys()[-1]

    node = find_winner(df)
    logger.info("Atrybut z największą wartością zysku informacyjnego: {}".format(node))

    # unikalne wartości dla wskazanego atrybutu
    att_values = np.unique(df[node])

    # utworzenie pustego słownika do przechowania drzewa
    if tree is None:
        tree = {}
        tree[node] = {}

    # wykonujemy pętlę aby zbudować drzewo, jeśli podzbiór jest czysty
    # zatrzymujemy działanie pętli
    for value in att_values:
        logger.info("Sprawdzamy wartość({}): {}".format(node, value))
        subtable = get_subtable(df, node, value)
        class_value, counts = np.unique(subtable['item_class'], return_counts=True)
        # print(class_value)
        if len(counts) == 1:
            logger.info("Zidentyfikowane jednoznacznie: {}".format(class_value[0]))
            tree[node][value] = class_value[0]
        else:
            logger.info("Wartość niepozwala na jednoznaczne określenie, sprawdzamy kolejny atrybut.")
            tree[node][value] = build_tree(subtable)

    logger.info("Zakończenie sprawdzania dla atrybutu: {}".format(node))

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


if __name__ == '__main__':

    logger.info("Czas startu: {:%Y-%m-%d %H:%M:%S}\n".format(datetime.now()))
    logger.info(
        "Witam w programie realizującym algorytm budowy drzewa decyzyjnego oraz testującym jego prawidłowość.\n")

    # przygotowanie klasy generującej dane z atrybutami na podstawie obrazów generowanych w przez opencv
    ds = DataSet()

    logger.info("Wygenerowanie obrazów testowych i zidenftyfikowanie niezbędnych predykatów.")
    training_dataset = ds.generate_training_dataset()
    # przekazanie ich do DataFrame w celu łatwiejszego operowania na nich oraz wizualizacji
    df_training = pd.DataFrame(training_dataset)

    logger.info("Tabela z danymi treningowymi:")
    logger.info(df_training.to_string())

    logger.info(
        "\nPrzekazuje dane do funkcji realizującej algorytm budowy drzewa decyzyjnego build_tree(df_training)\n")
    generated_tree = build_tree(df_training)
    logger.info("Wynikowe drzewo decyzyjne:")
    logger.info(json.dumps(eval(str(generated_tree)), sort_keys=True, indent=4))

    test_images_path = os.getcwd() + '\\obrazy\\testowe'
    logger.info("Podaj ścieżkę do obrazów testowych (*.jpg) lub zostaw puste aby użyć domyślnego katalogu.")
    logger.info("Domyślnie: [{}].".format(test_images_path))
    in_path = input()
    if not in_path:
        in_path = test_images_path

    logger.info("Użyta ścieżka: {}".format(in_path))
    test_dataset = ds.generate_dataset_from_dir(in_path)
    if test_dataset:
        # przekazanie ich do DataFrame w celu łatwiejszego operowania na nich oraz wizualizacji
        df_test = pd.DataFrame(test_dataset)

        logger.info("\nTabela z danymi treningowymi:")
        logger.info(df_test.to_string())

        logger.info("\nSprawdzamy działanie drzewa decyzyjnego na danych testowych:\n")
        for index, image_meta_data in df_test.iterrows():
            logger.info("Nazwa pliku: {}".format(image_meta_data.item_class))
            logger.info("Klasa zidentyfikowana przez drzewo decyzyjne: {}\n".format(
                predict(image_meta_data, generated_tree)
            ))
    logger.info("Czas końca: {:%Y-%m-%d %H:%M:%S}".format(datetime.now()))
    input()

