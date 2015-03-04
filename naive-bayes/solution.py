import csv
import nltk
import random
import time
import re

def get_length_bucket(text_length):
    if text_length > 138:
        return "true"
    return "false"


    
def capital_ratio(text):
    text = text.strip()
    text = text.replace(" ", "")
    upper = 0
    for char in text:
        if char.isupper():
            upper += 1
    ratio = float(upper) / float(len(text))
    if ratio > 0.8:
        return "caps"
    else:
        return "less"
        
# lower case and return words
def split_and_lower(text):
    lower = text.lower()
    words = lower.split()
    return words

def contains_std(text):
    words = split_and_lower(text)
    if "std" in words:
        return "true"
    return "false"

def contains_txt(text):
    words = split_and_lower(text)
    if "txt" in words:
        return "true"
    return "false"

def contains_claim(text):
    words = split_and_lower(text)
    if "claim" in words:
        return "true"
    return "false"

def contains_number(text):
    if bool(re.findall("\d{5}", text)):
        return "true"
    return "false"

def contains_currency(text):
    currency = ["$", "£", "€"]
    for sign in currency:
        if sign in text:
            return "true"
    return "false"
    

def text_features(text):
    return {
        "max_length": get_length_bucket(len(text)),
        "contains_std": contains_std(text),
        "contains_txt": contains_txt(text),
        "contains_claim": contains_claim(text),
        "contains_number": contains_number(text),
        "contains_currency": contains_currency(text),
        
    }

errors = []

def get_feature_sets():
    # open the file, which we've placed at /home/vagrant/repos/datasets/clean_twitter_data.csv
    # 'rb' means read-only mode and binary encoding
    f = open('/home/vagrant/repos/datasets/sms_spam_or_ham.csv', 'rb')

    # let's read in the rows from the csv file
    rows = []
    for row in csv.reader(f):
        rows.append(row)

    # now let's generate the output that we specified in the comments above
    output_data = []

    # let's just run it on 100,000 rows first, instead of all 1.5 million rows
    # when you experiment with the `twitter_features` function to improve accuracy
    # feel free to get rid of the row limit and just run it on the whole set
    for row in rows[:10000]:
        if len(row) != 2:
            continue
        # Remember that row[0] is the label, either 0 or 1
        # and row[1] is the tweet body

        # get the label
        label = row[0]

        # get the tweet body and compute the feature dictionary
        feature_dict = text_features(row[1])

        # add the tuple of feature_dict, label to output_data
        data = (feature_dict, label)
        output_data.append(data)

    # close the file
    f.close()
    return output_data


def get_training_and_validation_sets(feature_sets):
    """
    This takes the output of `get_feature_sets`, randomly shuffles it to ensure we're
    taking an unbiased sample, and then splits the set of features into
    a training set and a validation set.
    """
    # randomly shuffle the feature sets
    random.shuffle(feature_sets)

    # get the number of data points that we have
    count = len(feature_sets)
    # 20% of the set, also called "corpus", should be training, as a rule of thumb, but not gospel.

    # we'll slice this list 20% the way through
    slicing_point = int(.20 * count)

    # the training set will be the first segment
    training_set = feature_sets[:slicing_point]
    
    # data test segment for training

    # the validation set will be the second segment
    validation_set = feature_sets[slicing_point:]
    return training_set, validation_set


def run_classification(training_set, validation_set):
    # train the NaiveBayesClassifier on the training_set
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # let's see how accurate it was
    accuracy = nltk.classify.accuracy(classifier, validation_set)
    print "The accuracy was.... {}".format(accuracy)
    return classifier

def predict(classifier, new_text):
    """
    Given a trained classifier and a fresh data point (a tweet),
    this will predict its label, either 0 or 1.
    """
    return classifier.classify(text_features(new_text))


start_time = time.time()

print "Let's use Naive Bayes!"

our_feature_sets = get_feature_sets()
our_training_set, our_validation_set = get_training_and_validation_sets(our_feature_sets)
print "Size of our data set: {}".format(len(our_feature_sets))

print "Now training the classifier and testing the accuracy..."
classifier = run_classification(our_training_set, our_validation_set)

end_time = time.time()
completion_time = end_time - start_time
print "It took {} seconds to run the algorithm".format(completion_time)

# show the most informative (meaningful) features in the model, defaulting to showing the top 100.
# I explain below how to read these results
classifier.show_most_informative_features()