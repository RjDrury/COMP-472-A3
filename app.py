from helpers import *
from nb_bow import *


def main():
    all_claims, covid_claims, non_covid_claims = get_training_data_claims()

    original_vocab = get_original_vocabulary(all_claims)
    # print(original_vocab)

    filtered_vocab = get_filtered_vocabulary(all_claims)
    # print(filtered_vocab)

    print(len(original_vocab))
    print(len(filtered_vocab))

    # created a model object, needs to be trained and then used to predict
    model = nb_bow()



if __name__ == '__main__':
    main()