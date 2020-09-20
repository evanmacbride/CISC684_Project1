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

def printy(node):
    if node:
        print(node)
        if node.left:
            printy(node.left)
        if node.right:
            printy(node.right)
    return

# Get the label (i.e. attribute name) with the highest information gain.
#
# In this version, only use a Pandas dataframe. Don't use Attribute objects or
# a dict container.
def information_gain_heuristic(s, attr_list, df):
    gain_list = []
    for attr in attr_list:
        gain_list.append(information_gain(s, attr, df))
    return max(gain_list, key=lambda item:item[1])[0]

# Get the information gain for an attribute.
#
# This version uses Pandas directly, without using my Attribute
# class. In this version, s and attr are ONLY labels. Note that df (the Pandas
# dataframe) is a new parameter.
def information_gain(s, attr, df):
    s_pos = len(df.loc[(df[s] == 1)])
    s_neg = len(df.loc[(df[s] == 0)])
    s_total = len(df.index)
    attr_val0_pos = len(df.loc[(df[s] == 1) & df[attr] == 0])
    attr_val0_neg = len(df.loc[(df[s] == 0) & df[attr] == 0])
    attr_val0_total = len(df.loc[(df[attr] == 0)])
    attr_val1_pos = len(df.loc[(df[s] == 1) & df[attr] == 1])
    attr_val1_neg = len(df.loc[(df[s] == 0) & df[attr] == 1])
    attr_val1_total = len(df.loc[(df[attr] == 1)])
    attr_total = len(df.index)
    # The entropy of the entire set
    entropy_s = get_entropy(s, s_pos, s_neg, s_total)
    # The entropy of the subset that has "0" as a value for the attribute
    entropy_attr_0 = get_entropy(attr, attr_val0_pos, attr_val0_neg, attr_val0_total)
    # The entropy of the subset that has "1" as a value for the attribute
    entropy_attr_1 = get_entropy(attr, attr_val1_pos, attr_val1_neg, attr_val1_total)
    # Subtract from the set's entropy the entropy of each value multiplied by its proporition in the set
    gain = entropy_s[1] - (attr_val0_total/attr_total) * entropy_attr_0[1] - (attr_val1_total/attr_total) * entropy_attr_1[1]
    # Return the label of the attribute and its gain
    return attr, gain


def get_entropy(node_label, val1_instances, val0_instances, total):
    """The formula for entropy is Entropy = -p_1 * log_2(p_1) - p_0 * log_2(p_0)"""
    """As per Mitchell p. 56, 0 log 0 is defined as zero """
    try:
        entropy = (-1 * val1_instances/total * math.log2(val1_instances/total if val1_instances/total > 0 else 1)) - \
                  (val0_instances/total * math.log2(val0_instances/total if val0_instances/total > 0 else 1))
    # I'm getting an error where I have more attributes set to 0 or 1 than I
    # have total rows.
    except ZeroDivisionError as e:
        print("val1_instances: " + str(val1_instances))
        print("val0_instances: " + str(val0_instances))
        print("total: " + str(total))
        print("node: " + node_label)
        print(e)
        sys.exit()
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
        #A = random.choice(attributes_list) # For testing. Swap with info_gain
        A = information_gain_heuristic(target_attribute, attributes_list, examples_list)
        root.label = A
        for i in [0,1]:
            new_branch = Node()
            new_branch.label = str(i)
            examples_list_vi = examples_list.loc[examples_list[A] == i]
            if (examples_list_vi.empty):
                new_leaf = Node()
                new_leaf.label = examples_list[target_attribute].mode()[0]
                new_branch.insert(new_leaf)
                root.insert(new_branch)
            else:
                trimmed_attributes = copy.deepcopy(attributes_list)
                trimmed_attributes.remove(A)
                new_branch.insert(id3(examples_list_vi,target_attribute,trimmed_attributes))
                root.insert(new_branch)
    return root

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

tree = id3(train_df, "Class", list(train_df.columns[0:-1]))
#tree = id3(micro_df, "Class", list(micro_df.columns[0:-1]))
print(tree)
