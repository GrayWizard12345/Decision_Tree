import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Decision tree classification model.')
    parser.add_argument('train', metavar='train_file', type=str, nargs='+',
                        help='Path to a training dataset file')
    parser.add_argument('test', metavar='test_file', type=str, nargs='+',
                        help='path to a test set file')
    parser.add_argument('result', metavar='output_file_name', type=str, nargs='+',
                        help='Output file name')

    return parser.parse_args()


def read_data(file_name):
    """
    :param file_name: file name to read the dataset from.
    :return: a tuple containing a 2d array of values and the list of attribute names
    """
    try:
        f = open(file_name)
        dataset = []

        line = f.readline()
        true_attrs = line.strip("\n").split("\t")
        attr_names = true_attrs[:-1]

        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip("\n").split("\t")
            dataset.append(line)
        f.close()

        return dataset, attr_names, true_attrs
    except FileNotFoundError as e:
        print(e)


def unique_attr_vals(dataset, attr_index):
    """
    :param dataset: a 2d array containing the dataset
    :param attr_index: index of the attribute (column number)
    :return: a set of unique values
    """
    return set([row[attr_index] for row in dataset])


def class_counts(dataset):
    """
    :param dataset: a 2d array containing the dataset
    :return: a dictionary containing the count of each class (class is a key)
    """
    counts = {}
    for row in dataset:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def gini_index(dataset):
    classes = class_counts(dataset)
    s = 0
    for label in classes:
        prob = classes[label] / len(dataset)
        s += prob * prob

    return 1 - s


def impurity_reduction(current_impurity: float, splits: list):
    tot_len = 0
    for split in splits:
        tot_len += len(split)

    s = 0
    for split in splits:
        weight = len(split) / tot_len
        s += weight * gini_index(split)

    return current_impurity - s


class BranchingCondition:

    def __init__(self, attr_name, attr_index, value):
        self.attr_name = attr_name
        self.attr_index = attr_index
        self.value = value

    def test(self, row):
        return row[self.attr_index] == self.value

    def __str__(self):
        return self.attr_name + " - " + self.value



def split_on_condition(dataset, condition: BranchingCondition):
    satisfied_rows = []
    unsatisfied_rows = []
    for row in dataset:
        if condition.test(row):
            satisfied_rows.append(row)
        else:
            unsatisfied_rows.append(row)

    return satisfied_rows, unsatisfied_rows


def best_split(dataset, attr_names):
    highest_impurity_reduction = 0
    best_branching_condition = None
    current_impurity = gini_index(dataset)

    for i in range(len(attr_names)):

        unique_attribute_values = unique_attr_vals(dataset, i)

        for val in unique_attribute_values:

            condition = BranchingCondition(attr_names[i], i, val)

            satisfied_rows, unsatisfied_rows = split_on_condition(dataset, condition)
            if len(satisfied_rows) == 0 or len(unique_attribute_values) == 0:
                continue

            imp_red = impurity_reduction(current_impurity, [satisfied_rows, unsatisfied_rows])

            if imp_red >= highest_impurity_reduction:
                best_branching_condition = condition
                highest_impurity_reduction = imp_red

    return best_branching_condition, highest_impurity_reduction


def write_data(file_name, dataset):
    """
    :param file_name: file name to read the dataset from.
    :return: a tuple containing a 2d array of values and the list of attribute names
    """
    try:
        f = open(file_name, "w+")

        sep = "\t"
        for datum in dataset:
            line = sep.join(datum)
            line += "\n"
            f.write(line)

    except FileNotFoundError as e:
        print(e)