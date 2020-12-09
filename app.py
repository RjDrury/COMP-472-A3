from helpers import *
from nb_bow import *


def main():
    #all_claims, covid_claims, non_covid_claims = get_training_data_claims()
    model = nb_bow()
    feat = [{'I':4, 'love':1, 'turnip':0}, {'I':1, 'love':0, 'turnip':9}, {'I':2, 'love':4, 'turnip':6}]
    lab = ['yes', 'yes', 'no']
    model.train(feat, lab)

    '''original_vocab = get_original_vocabulary(all_claims)
    # print(original_vocab)

    filtered_vocab = get_filtered_vocabulary(all_claims)
    # print(filtered_vocab)

    print(len(original_vocab))
    print(len(filtered_vocab))'''



if __name__ == '__main__':
    main()