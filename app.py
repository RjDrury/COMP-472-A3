from constants import *
import csv


def main():
    tsv_file = open(train_set)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    included_cols = [0, 1, 2]
    content = []

    for row in read_tsv:
        content.append(list(row[i] for i in included_cols))

    for row in content:
        print(row)

    tsv_file.close()



if __name__ == '__main__':
    main()