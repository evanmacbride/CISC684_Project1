import sys
import pandas as pd
from id3_algorithm import *

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

# perform ID3 algorithm
# First, get the attribute label, total number of 1s, 0s, and the total number of values for each column
#attribute_labels, attribute_positives, attribute_negatives, attribute_totals = get_attribute_labels(train_df)

attribute_labels, attribute_val1s, attribute_val0s, attribute_totals = get_attribute_labels(train_df)

# Next, get the output variable's (Class variable) data
class_label = attribute_labels[len(attribute_labels) - 1]
positive_class_instances = attribute_val1s[len(attribute_val1s) - 1]
negative_class_instances = attribute_val0s[len(attribute_val0s) - 1]
total = attribute_totals[len(attribute_totals) - 1]

for i in range(0, len(attribute_labels)):
    print(attribute_labels[i], ", 1: ", attribute_val1s[i], ", 0: ",
          attribute_val0s[i], ", Total:", attribute_totals[i])

# For every attribute:
#   a.) Calculate entropy
entropy_of_attributes = []
for i in range(len(attribute_labels)):
    entropy_of_attributes.append(get_entropy(attribute_labels[i], attribute_val1s[i],
                                             attribute_val0s[i], attribute_totals[i]))

# Get the entropy for the Class
class_entropy = entropy_of_attributes[len(entropy_of_attributes) - 1]

#   b.) Take average information entropy for the current attribute
#   c.) Calculate gain for the current attribute


# if to_print == 'yes':
#     print_tree()

# Printouts for Evan...
print("Class: ", class_label, "Positive: ", positive_class_instances,
      "Negative: ", negative_class_instances, "Total", total, "\n")

print("Attr", "Entropy\n")

for attr, entr in entropy_of_attributes:
    print(attr, ": ", entr)
### End of my printouts
