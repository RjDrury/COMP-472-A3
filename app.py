from helpers import *
from nb_bow import *


def main():
    index_to_content_map, index_to_id_map, index_to_validity_map = get_training_data_claims()

    original_vocab = get_original_vocabulary(index_to_content_map)
    filtered_vocab = get_filtered_vocabulary(index_to_content_map)

    # print(len(original_vocab))
    # print(len(filtered_vocab))

    original_dict = get_list_of_dictionaries(index_to_content_map, original_vocab)
    filtered_dict = get_list_of_dictionaries(index_to_content_map, filtered_vocab)
    # created a model object, needs to be trained and then used to predict
    model = nb_bow()
    model.train(filtered_dict, index_to_validity_map)

    # get output and predict

    test_claims, index_to_id_map, index_to_validity_map = get_testing_data_claims()
    dict_for_predictions = get_list_of_dict_for_predictions(test_claims)
    predictions = model.predict(dict_for_predictions)
    print(predictions)

if __name__ == '__main__':
    main()