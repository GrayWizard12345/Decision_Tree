from utils import *
from tree import Node, Leaf


def build_tree(dataset, attr_names):
    condition, imp_red = best_split(dataset, attr_names)

    if imp_red == 0:
        return Leaf(dataset)

    left_rows, right_rows = split_on_condition(dataset, condition)

    left_branch = build_tree(left_rows, attr_names)
    right_branch = build_tree(right_rows, attr_names)

    return Node(condition, left_branch, right_branch)


def classify(touple, node):
    if isinstance(node, Leaf):
        return node

    if node.condition.test(touple):
        return classify(touple, node.left)
    else:
        return classify(touple, node.right)


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    train_f = args.train[0].strip()
    test_f = args.test[0].strip()
    result_f = args.result[0].strip()

    dataset, attr_names, atrs = read_data(train_f)

    print(attr_names)

    unique_attrs = {}
    for i in range(len(attr_names)):
        unique_attrs[attr_names[i]] = unique_attr_vals(dataset, i)

    classes = class_counts(dataset)

    # print("\nUnique Attributes:")
    # print(unique_attrs)
    #
    # print("\nClasses:")
    # print(classes, len(classes))
    #
    # print("\nGini(D):")
    # print(gini_index(dataset))

    root = build_tree(dataset, attr_names)

    test, _, _ = read_data(test_f)

    for touple in test:
        label = classify(touple, root).prediction_label
        touple.append(label)

    print(atrs)
    print(_)
    test.insert(0, atrs)

    write_data(result_f, test)