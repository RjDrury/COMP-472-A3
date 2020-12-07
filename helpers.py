from constants import *
import csv
import re


def get_training_data_claims():
    tsv_file = open(train_set)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    train_claims, covid_claims, non_covid_claims = get_claims_lists(read_tsv)
    tsv_file.close()

    return train_claims, covid_claims, non_covid_claims


def get_testing_data_claims():
    tsv_file = open(test_set)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    test_claims, covid_claims, non_covid_claims = get_claims_lists(read_tsv)
    tsv_file.close()

    return test_claims, covid_claims, non_covid_claims


def get_claims_lists(read_tsv):
    included_cols = [1, 2]  # 0 = content & 1 = covid flags
    all_data = []

    for row in read_tsv:
        all_data.append(list(row[i] for i in included_cols))

    # content from all covid claims
    covid_claims = [i[0].lower() for i in all_data if i[1] == 'yes']

    # content from all non covid claims
    non_covid_claims = [i[0].lower() for i in all_data if i[1] == 'no']

    # content from all claims
    all_claims = [i[0].lower() for i in all_data]

    return all_claims, covid_claims, non_covid_claims


def get_original_vocabulary(documents_list):
    all_words = []
    for string in documents_list:
        word_list = re.sub("[^\w]", " ", string).split()
        for word in word_list:
            if word not in all_words:
                all_words.append(word)

    return all_words


def get_filtered_vocabulary(documents_list):
    present_once = []
    present_more_than_once = []

    for string in documents_list:
        word_list = re.sub("[^\w]", " ", string).split()
        for word in word_list:
            if word not in present_once:
                present_once.append(word)
            elif word not in present_more_than_once:
                present_more_than_once.append(word)

    return present_more_than_once

