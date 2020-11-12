from utils import *
from tree import Node, Leaf


def build_tree(dataset, attr_names):
    """
    Recursively build the tree in the top to down manner.

    Looks for the best split attribute on current node.
    If the reduction on impurity after split is 0, then return a Leaf node.

    Otherwise split using the best condition calculated.
    Call build_tree on its left child and then on its right child.

    :param dataset: a 2d list representing a list of data objects :param attr_names: a list containing attribute
    names
    :return: a node with left child as left_branch and right child as right_branch and condition as the best
    splitting condition on this node.
    """
    condition, imp_red = best_split(dataset, attr_names)

    if imp_red == 0:
        return Leaf(dataset)

    left_rows, right_rows = split_on_condition(dataset, condition)

    left_branch = build_tree(left_rows, attr_names)
    right_branch = build_tree(right_rows, attr_names)

    return Node(condition, left_branch, right_branch)


def classify(data_object, node):
    """
    Recursively iterates the Decision tree until the leaf node is reached.

    :param data_object: a row from the test set
    :param node: a node that is being passed by the data_object
    :return: a Leaf node reached.
    """
    if isinstance(node, Leaf):
        return node

    if node.condition.test(data_object):
        return classify(data_object, node.left)
    else:
        return classify(data_object, node.right)


if __name__ == '__main__':

    # Parse the CLI arguments
    args = parse_arguments()
    print(args)
    train_f = args.train[0].strip()
    test_f = args.test[0].strip()
    result_f = args.result[0].strip()

    # Read data from the training file
    dataset, attr_names, atrs = read_data(train_f)

    # First call to build tree to get the root of the tree
    root = build_tree(dataset, attr_names)

    # Read test dataset
    test, _, _ = read_data(test_f)

    # Make a prediction for each row in the dataset
    for data_object in test:
        label = classify(data_object, root).prediction_label
        data_object.append(label)

    # Insert the attribute names and the class name to the datasets begining
    test.insert(0, atrs)

    # Write data to the result_f file
    write_data(result_f, test)