import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
random.seed(42)

#my imports
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



#HELPER FUNCTIONS

#tokenizing
def get_words(text):
    words = []
    dict_counter = {}
    words_lower = text.lower() 
    words_split = words_lower.split(' ')
    for word in words_split:
        if word.isalpha():
            words.append(word)
            #print(words)
        
    return words
            

#counting words + deleting words occurring less than x times    
def word_counter(samples):
    final_list = []
    for item in samples: #samples = list of strings // item = single string
        word_count = {}
        words = get_words(item)
        for word in words:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
        del_words = []
        for word, count in word_count.items(): #Iterate the word_count dict
            if (count < 3): # If the word occurs less than x times
                del_words.append(word) # Add it to a list to be deleted. We cannot delete it here because of how the it

        for word in del_words:
            del word_count[word]

        final_list.append(word_count)
    #print(f'final list: {final_list}')
    return final_list

#END OF HELPER FUNCTIONS
            

###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X


def extract_features(samples):
    counter = word_counter(samples) 
    feature_vector = DictVectorizer()
    ndarray = feature_vector.fit_transform(counter).toarray()
    #print(ndarray)
    return ndarray


##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):
    pca = PCA(n_components=n)
    dim_red = pca.fit_transform(X)
    #print(dim_red)
    return dim_red

    #svd = TruncatedSVD(n_components=n)
    #dim_red = svd.fit_transform(X)
    #print(dim_red)
    #return dim_red



##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = DecisionTreeClassifier() # <--- REPLACE THIS WITH A SKLEARN MODEL
    elif clf_id == 2:
        clf = GaussianNB() # <--- REPLACE THIS WITH A SKLEARN MODEL
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf


#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evalute model
    print("Evaluating classcifier ...")
    evalute_classifier(clf, X_test, y_test)


#returning 80/20 train test split in four lists
def shuffle_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    return X_train, X_test, y_train, y_test


#training the classifier on the training data
def train_classifer(clf, X, y):
    assert is_classifier(clf)
    model = clf.fit(X,y)


# getting accuracy, precision, recall, and F-measure of trained classifier  
def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    clf_prediction = clf.predict(X)
    accuracy = metrics.accuracy_score(clf_prediction, y)
    print('Accuracy:', accuracy)
    print()
    print(metrics.classification_report(clf_prediction, y))



######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=args.model_id, 
            n_dim=args.number_dim_reduce
            )
