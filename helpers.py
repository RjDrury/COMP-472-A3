from constants import *
import csv


def get_training_data_claims():
    tsv_file = open(train_set)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    included_cols = [0, 1, 2]
    train_claims = []

    # all_claims[0] = ids
    # all_claims[1] = content
    # all_claims[2] = covid flags
    for row in read_tsv:
        train_claims.append(list(row[i] for i in included_cols))

    # content from all covid claims
    covid_claims = [i[1] for i in train_claims if i[2] == 'yes']

    # content from all non covid claims
    non_covid_claims = [i[1] for i in train_claims if i[2] == 'no']

    tsv_file.close()

    return train_claims, covid_claims, non_covid_claims


def get_testing_data_claims():
    tsv_file = open(test_set)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    included_cols = [0, 1, 2]
    test_claims = []

    # all_claims[0] = ids
    # all_claims[1] = content
    # all_claims[2] = covid flags
    for row in read_tsv:
        test_claims.append(list(row[i] for i in included_cols))

    # content from all covid claims
    covid_claims = [i[1] for i in test_claims if i[2] == 'yes']

    # content from all non covid claims
    non_covid_claims = [i[1] for i in test_claims if i[2] == 'no']

    tsv_file.close()

    return test_claims, covid_claims, non_covid_claims

