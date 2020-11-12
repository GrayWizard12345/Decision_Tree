from utils import class_counts


class Node:

    def __init__(self, condition, left, right):
        self.condition = condition
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.condition)


class Leaf:

    def __init__(self, dataset):
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
        return self.prediction_label + ":" + str(self.prediction_confidence)
