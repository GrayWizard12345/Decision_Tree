from utils import class_counts


class Node:

    def __init__(self, condition, left, right):
        """
        :param condition: an attribute value upon which the splitting on the node occurs
        :param left: data tuples that satisfy the condition go to the left node of the tree
        :param right: other tupels go to the right
        """
        self.condition = condition
        self.left = left
        self.right = right


class Leaf:

    def __init__(self, dataset):
        """
        A leaf node that makes determine the label
        """
        possible_classes = class_counts(dataset)
        mx = 0
        l = None
        s = 0

        for label in possible_classes:
            s += possible_classes[label]

        for label in possible_classes:

            if possible_classes[label] >= mx:
                mx = possible_classes[label]
                l = label

        self.prediction_label = l
        self.prediction_confidence = mx / s


    def __str__(self):
        """
        String representation of the class
        """
        return self.prediction_label + " with confidence " + str(self.prediction_confidence)
