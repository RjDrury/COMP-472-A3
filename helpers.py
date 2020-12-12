from constants import *
import csv
import re
from sklearn import metrics

def get_training_data_claims():
    tsv_file = open(train_set, encoding="utf8")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    train_claims, index_to_id_map, index_to_validity_map = get_claims_lists(read_tsv)
    tsv_file.close()

    return train_claims, index_to_id_map, index_to_validity_map


def get_testing_data_claims():
    tsv_file = open(test_set, encoding="utf8")
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
        dict = {}
        for word in vocab:
            amount = count_occurrences(word, string)
            dict[word] = amount

        list_of_dict.append(dict)

    return list_of_dict

def get_list_of_dict_for_predictions(content_list):
    list_of_dict = []
    for string in content_list:
        dict = {}
        words = string.split(" ")
        for word in words:
            amount = count_occurrences(word, string)
            dict[word] = amount

        list_of_dict.append(dict)

    return list_of_dict

def count_occurrences(word, sentence):
    return sentence.split().count(word)

def write_trace_file(index_to_id_map, predictions, predition_percentage, index_to_validity_map):
    trace_file = open("output/"+"trace_NB-BOW-OV.txt", "w")

    for i in range(len(index_to_id_map)):
        label = "wrong"
        if predictions[i] == index_to_validity_map[i]:
            label = "correct"
        trace_file.write(index_to_id_map[i] + "  " +  predictions[i] + "  " + str(predition_percentage[i])+ "  " + index_to_validity_map[i] + " " + label +"\n")

def write_eval_file(predictions,index_to_validity_map):
    eval_file = open("output/"+"eval_NB-BOW-OV.txt", "w")
    #s = metrics.classification_report(index_to_validity_map, predictions)
    #trace_file.write(s)

    accuracy = metrics.accuracy_score(index_to_validity_map,predictions)

    yes_pres_score = metrics.precision_score(index_to_validity_map,predictions, pos_label="yes")
    no_pres_score = metrics.precision_score(index_to_validity_map,predictions, pos_label="no")

    yes_recall_score = metrics.recall_score(index_to_validity_map,predictions, pos_label="yes")
    no_recall_score = metrics.recall_score(index_to_validity_map,predictions, pos_label="no")

    yes_f1_score = metrics.f1_score(index_to_validity_map,predictions, pos_label="yes")
    no_f1_score = metrics.f1_score(index_to_validity_map,predictions, pos_label="no")

    eval_file.write(str(accuracy)+"\n")
    eval_file.write(str(yes_pres_score) + " " +  str(no_pres_score) + "\n")
    eval_file.write(str(yes_recall_score) + " " +   str(no_recall_score) + "\n")
    eval_file.write(str(yes_f1_score) + " " +   str(no_f1_score) + "\n")