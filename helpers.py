from constants import *
import csv
import re


def get_training_data_claims():
    tsv_file = open(train_set)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    train_claims, index_to_id_map, index_to_validity_map = get_claims_lists(read_tsv)
    tsv_file.close()

    return train_claims, index_to_id_map, index_to_validity_map


def get_testing_data_claims():
    tsv_file = open(test_set)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    test_claims, index_to_id_map, index_to_validity_map = get_claims_lists(read_tsv)
    tsv_file.close()

    return test_claims, index_to_id_map, index_to_validity_map


def get_claims_lists(read_tsv):
    included_cols = [0, 1, 2]  # 0 = content & 1 = covid flags
    all_data = []

    for row in read_tsv:
        all_data.append(list(row[i] for i in included_cols))

    index_to_id_map = [i[0] for i in all_data]
    index_to_content_map = [i[1].lower() for i in all_data]
    index_to_validity_map = [i[2] for i in all_data]

    return index_to_content_map[1:], index_to_id_map[1:], index_to_validity_map[1:]


def get_original_vocabulary(documents_list):
    all_words = []
    for string in documents_list:
        for word in string.split():
            if word not in all_words:
                all_words.append(word)

    return all_words


def get_filtered_vocabulary(documents_list):
    present_once = []
    present_more_than_once = []

    for string in documents_list:
        for word in string.split():
            if word not in present_once:
                present_once.append(word)
            elif word not in present_more_than_once:
                present_more_than_once.append(word)

    return present_more_than_once


def get_list_of_dictionaries(content_list, vocab):
    list_of_dict = []
    for string in content_list:
        dict = []
        for word in vocab:
            amount = count_occurrences(word, string)
            dict.append({word: amount})

        list_of_dict.append(dict)

    return list_of_dict


def count_occurrences(word, sentence):
    return sentence.split().count(word)
