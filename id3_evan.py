import pandas as pd
import sys
import math
import copy
import random # for testing only

class Node:
    def __init__(self, label=None):
        self.left = None
        self.right = None
        self.label = label

    def __repr__(self):
        return '{label:' + str(self.label) + '}'

    def __str__(self):
        return str(self.label)

    def insert(self, label):
        if self.left is None:
            self.left = Node(label)
        elif self.right is None:
            self.right = Node(label)
        else:
            sys.stderr.write('Error: Cannot insert. All branches at node are set.\n')
            sys.exit()

    def printout(self, node):
        tree = ''
        if node:
            tree += str(node.label)
            if (self.left):
                tree += self.printout(node.left)
            if (self.right):
                tree += self.printout(node.right)
        return tree

    def __str__(self, offset=''):
        """ TODO: Get this working properly. """
        """
        #print(offset + str(self.label))
        #print(self.label)
        if self.left:
            #offset = offset + "-"
            print(self.left, offset)
        elif self.right:
            #offset = offset + "-"
            print(self.right, offset)
        """
        return str(self.label)

class Attribute:
    def __init__(self, label):
        self.label = label
        self.val0_pos = 0   # How many positive examples a "0" value appears in
        self.val0_neg = 0   # How many negative examples a "0" value appears in
        self.val1_pos = 0   # How many positive examples a "1" value appears in
        self.val1_neg = 0   # How many negative examples a "1" value appears in

    def getTotalVal0(self):
        return self.val0_pos + self.val0_neg

    def getTotalVal1(self):
        return self.val1_pos + self.val1_neg

    def getTotal(self):
        return self.getTotalVal0() + self.getTotalVal1();

def printy(node):
    if node:
        print(node)
        if node.left:
            printy(node.left)
        if node.right:
            printy(node.right)
    return

# Get the label (i.e. attribute name) that has the highest information gain
def information_gain_heuristic(s, attr_dict):
    gain_list = []
    for attr in attr_dict:
        gain_list.append(information_gain(s, attr_dict[attr]))
    return max(gain_list, key=lambda item:item[1])[0]

def information_gain(s, attr):
    # The entropy of the entire set
    entropy_s = get_entropy(s.label, s.getTotalVal1(), s.getTotalVal0(), s.getTotal())
    # The entropy of the subset that has "0" as a value for the attribute
    entropy_attr_0 = get_entropy(attr.label, attr.val0_pos, attr.val0_neg, attr.getTotalVal0())
    # The entropy of the subset that has "1" as a value for the attribute
    entropy_attr_1 = get_entropy(attr.label, attr.val1_pos, attr.val1_neg, attr.getTotalVal1())
    # Subtract from the set's entropy the entropy of each value multiplied by its proporition in the set
    gain = entropy_s[1] - (attr.getTotalVal0()/attr.getTotal()) * entropy_attr_0[1] - (attr.getTotalVal1()/attr.getTotal()) * entropy_attr_1[1]
    # Return the label of the attribute and its gain
    return attr.label, gain

def get_entropy(node_label, val1_instances, val0_instances, total):
    """The formula for entropy is Entropy = -p_1 * log_2(p_1) - p_0 * log_2(p_0)"""
    """As per Mitchell p. 56, 0 log 0 is defined as zero """
    entropy = (-1 * val1_instances/total * math.log2(val1_instances/total if val1_instances/total > 0 else 1)) - \
              (val0_instances/total * math.log2(val0_instances/total if val0_instances/total > 0 else 1))
    return node_label, entropy

# examples_list     A pandas dataframe
# target_attribute  The name of the target attribute (i.e. "Class")
# attributes_list   A list of attribute names (i.e. "XC", "XD", etc.)
def id3(examples_list, target_attribute, attributes_list):
    root = Node()
    A = ''
    # If all examples are positive, set root's label to "1"
    if (all(l == 1 for l in list(examples_list[target_attribute]))):
        root.label = 1
    # Elif all examples are negative, set root's label to "0"
    elif (all(l == 0 for l in list(examples_list[target_attribute]))):
        root.label = 0
    # Elif attributes_list is empty, set root's label to most common value of
    #   target attribute in examples
    elif (not attributes_list):
        root.label = examples_list[target_attribute].mode()[0]
    else:
        A = random.choice(attributes_list) # For testing. Swap with info_gain
        #attr_dict = get_example_data(examples_list)
        #class_attr = attr_dict[target_attribute]
        #del attr_dict[target_attribute]
        #A = information_gain_heuristic(class_attr, attr_dict)
        root.label = A
        for i in [0,1]:
            '''
            new_node = Node()
            examples_list_vi = examples_list.loc[examples_list[A] == i]
            if (examples_list_vi.empty):
                new_node.label = examples_list[target_attribute].mode()[0]
                root.insert(new_node)
            else:
                root.label = A
                trimmed_attributes = copy.deepcopy(attributes_list) # Do I really need deepcopy?
                trimmed_attributes.remove(A)
                root.insert(id3(examples_list_vi,target_attribute,trimmed_attributes))
            '''
            new_branch = Node()
            new_branch.label = str(i)
            examples_list_vi = examples_list.loc[examples_list[A] == i]
            if (examples_list_vi.empty):
                new_leaf = Node()
                new_leaf.label = examples_list[target_attribute].mode()[0]
                new_branch.insert(new_leaf)
                root.insert(new_branch)
            else:
                trimmed_attributes = copy.deepcopy(attributes_list) # Do I really need deepcopy?
                trimmed_attributes.remove(A)
                new_branch.insert(id3(examples_list_vi,target_attribute,trimmed_attributes))
                root.insert(new_branch)
    return root

def get_example_data(dataset):
    column_names = dataset.columns
    class_name = column_names[-1]
    classification = dataset[class_name]
    attr_dict = {}
    for col in column_names:
        current_column = dataset[col]
        attr_dict[col] = Attribute(col)
        attr_dict[col].label = col
        for i in range(len(current_column)):
            value = current_column[i]
            if (value == 0 and classification[i] == 0):
                count = attr_dict[col].val0_neg
                attr_dict[col].val0_neg = count + 1
            elif (value == 0 and classification[i] == 1):
                count = attr_dict[col].val0_pos
                attr_dict[col].val0_pos = count + 1
            elif (value == 1 and classification[i] == 0):
                count = attr_dict[col].val1_neg
                attr_dict[col].val1_neg = count + 1
            elif (value == 1 and classification[i] == 1):
                count = attr_dict[col].val1_pos
                attr_dict[col].val1_pos = count + 1
            else:
                sys.stderr.write('ILLEGAL VALUE FOUND IN CLASS DATA\n')
                sys.exit()
    return attr_dict

# get command line arguments ************ UNCOMMENT AFTER FINISHED DEVELOPING *********
# l = sys.argv[0]
# k = sys.argv[1]
# training_set = sys.argv[2]
# validation_set = sys.argv[3]
# test_set = sys.argv[4]
# to_print = sys.argv[5]

# l = 1
# k = 0
training_set = 'data_sets2/data_sets2/training_set.csv'
validation_set = 'data_sets2/data_sets2/validation_set.csv'
test_set = 'data_sets2/data_sets2/test_set.csv'
to_print = 'yes'

# run id3 algorithm

# convert training_set, validation_set, and test_set to dataframes
train_df = pd.read_csv(training_set)
validation_df = pd.read_csv(validation_set)
test_df = pd.read_csv(test_set)

micro_set = 'data_sets2/data_sets2/micro_set.csv'
micro_df = pd.read_csv(micro_set)

#attribute_labels_list, val1_instances_list, val0_instances_list, total_instances_list = get_attribute_labels(train_df)
#print(attribute_labels_list)
#print(val1_instances_list)
#print(train_df)

"""
attr_dict = get_example_data(train_df)
# Separate the class attribute
class_attr = attr_dict["Class"]
del attr_dict["Class"]
gain_list = []
for attr in attr_dict:
    print("%s: 0+:%i  0-:%i  1+:%i  1-:%i  Gain: %f" % (attr_dict[attr].label,
          attr_dict[attr].val0_pos, attr_dict[attr].val0_neg,
          attr_dict[attr].val1_pos, attr_dict[attr].val1_neg,
          information_gain(class_attr, attr_dict[attr])[1]))
    gain_list.append(information_gain(class_attr, attr_dict[attr]))
for label, gain in gain_list:
    print(label + ": " + str(gain))
#print(max(gain_list, key=lambda item:item[1]))
print(information_gain_heuristic(class_attr, attr_dict))
"""

#tree = id3(train_df, "Class", list(train_df.columns[0:-1]))
tree = id3(micro_df, "Class", list(micro_df.columns[0:-1]))
#print(tree.printout(tree))
print(tree)
