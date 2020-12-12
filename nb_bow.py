import numpy as np
import math

class nb_bow():
    def __init__(self):
        self.prior_probs = {'yes':0, 'no':0}
        self.conditional_yes = {}
        self.conditional_no = {}
        self.num_yes = 0
        self.num_no = 0
        self.smooth_numerator = 0.01  # constant
        self.smooth_denominator = 0  # set based on number of instances

    # reformulate training features into words per class for training
    def sort_by_class(self, train_features, train_labels):
        vocab = train_features[0].keys()
        z = np.zeros(len(vocab), dtype=int)
        vocab_dict_yes = dict(zip(vocab, z))
        vocab_dict_no = dict(zip(vocab, z))

        for i in range(len(train_labels)):
            tweet_words = np.column_stack((list(train_features[i].keys()), list(train_features[i].values())))
            tweet_words = np.flip(tweet_words[tweet_words[:, 1].argsort()], axis=0)
            if train_labels[i] == 'yes':
                for j in range(len(tweet_words)):
                    if int(tweet_words[j][1]) != 0:
                        vocab_dict_yes[tweet_words[j][0]] += int(tweet_words[j][1])
                    else:
                        break
            elif train_labels[i] == 'no':
                for j in range(len(tweet_words)):
                    if int(tweet_words[j][1]) != 0:
                        vocab_dict_no[tweet_words[j][0]] += int(tweet_words[j][1])
                    else:
                        break
        return vocab_dict_yes, vocab_dict_no

    # sets dictionary of prior probabilities (no smoothing required)
    def set_prior_probs(self, train_labels):
        total = len(train_labels)
        for i in train_labels:
            if i == 'yes':
                self.num_yes += 1
        self.num_no = total - self.num_yes
        self.prior_probs['yes'] = self.num_yes / total
        self.prior_probs['no'] = self.num_no / total

    # sets the parameter to add to the denominator to apply smoothing
    def set_smoothing_params(self, num_vocab):
        self.smooth_denominator = self.smooth_numerator * num_vocab

    # sets dictionaries of conditional probabilities for yes and no classes of each word (smoothing applied)
    def set_conditional_probs(self, words_yes, words_no):
        vocab = words_yes.keys()
        self.set_smoothing_params(len(vocab))
        conditional_prob_yes = []
        conditional_prob_no = []

        for i in vocab:
            conditional_prob_yes.append((words_yes[i] + self.smooth_numerator) / (self.num_yes + self.smooth_denominator))
            conditional_prob_no.append((words_no[i] + self.smooth_numerator) / (self.num_no + self.smooth_denominator))
        self.conditional_yes = dict(zip(vocab, conditional_prob_yes))
        self.conditional_no = dict(zip(vocab, conditional_prob_no))

    def train(self, train_features, train_labels):
        words_yes, words_no = self.sort_by_class(train_features, train_labels)
        self.set_prior_probs(train_labels)
        self.set_conditional_probs(words_yes, words_no)


    def predict(self, predict_features):
        predictions = []
        final_score = []
        sum_yes = 0
        sum_no = 0
        for i in range(len(predict_features)):
            tweet_words = np.column_stack((list(predict_features[i].keys()), list(predict_features[i].values())))
            tweet_words = np.flip(tweet_words[tweet_words[:, 1].argsort()], axis=0)
            for j in range(len(tweet_words)):
                try:
                    if int(tweet_words[j][1]) != 0:
                        sum_yes += math.log10(self.conditional_yes[tweet_words[j][0]]) * int(tweet_words[j][1])
                        sum_no += math.log10(self.conditional_no[tweet_words[j][0]]) * int(tweet_words[j][1])
                    else:
                        break
                except KeyError:
                    continue

            score_yes = math.log10(self.prior_probs['yes']) + sum_yes
            score_no = math.log10(self.prior_probs['no']) + sum_no
            if score_no > score_yes:
                predictions.append('no')
                final_score.append(score_no)
            else:
                predictions.append('yes')
                final_score.append(score_yes)
        return predictions, final_score
