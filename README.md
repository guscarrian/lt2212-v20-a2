# LT2212 V20 Assignment 2


**PART 1: Creating the feature table**

Helper functions:
- get_words: This function returns a list of all tokenized words from the corpus. It gets rid of punctuation and numbers, puts all characters in lowercase and splits the string on whitespace. 

- word_counter: It iterates through a list of strings, it counts each word's occurrence and creates a dictionary with the word as the key and the number of times it appears being the value. Also, all words that occur less than 3 times are removed from the list. It makes the data smaller (from 59540 to 10252), which helps us running the script faster in part 3. Finally, the dictionary is appended to a final list.


In extract_features, I have used DictVectorizer to transform the list of feature-value mappings to vectors.



**PART 2: Dimensionality reduction**

For the dimensionality reduction, I have used principal component analysis (PCA). 



**PART 3: Classify and evaluate**

- Model 1: Decision Tree Classifier
- Model 2: Naive Bayes Classifier



**PART 4: Try and discuss**

Unreduced features: 10252

|               | Accuracy | Precision | Recall | F-measure |
|---------------|----------|-----------|--------|-----------|
| Decision Tree | 0.23     | 0.25      | 0.21   | 0.22      |
| Naive Bayes   | 0.34     | 0.57      | 0.34   | 0.29      |



50% feature reduction = 5126; 
25% feature reduction = 2563;
10% feature reduction = 1026;
5% feature reduction = 513



Reduced features using PCA:
- Decision Tree Classifier:

|     | Accuracy | Precision | Recall | F-measure |
|-----|----------|-----------|--------|-----------|
| 50% | 0.15     | 0.17      | 0.15   | 0.15      |
| 25% | 0.16     | 0.20      | 0.16   | 0.17      |
| 10% | 0.16     | 0.19      | 0.16   | 0.16      |
| 5%  | 0.15     | 0.18      | 0.15   | 0.16      |



- Naive Bayes Classifier:

|     | Accuracy | Precision | Recall | F-measure |
|-----|----------|-----------|--------|-----------|
| 50% | 0.12     | 0.47      | 0.11   | 0.14      |
| 25% | 0.11     | 0.42      | 0.11   | 0.13      |
| 10% | 0.12     | 0.46      | 0.12   | 0.16      |
| 5%  | 0.097    | 0.59      | 0.10   | 0.12      |



Regarding the results obtained for the unreduced features, I was surprised to get such a bad performance. Although the accuracy level is quite low for both classifiers, it seems that Naive Bayes is doing slighly better not only in terms of accuracy but also in precision, recall and F-measure. 

When it comes to reduced features, there is a small difference within the results compared to the unreduced results. It could be said that applying dimensionality reduction lowers accuracy. Comparing both classifiers and looking at accuracy specifically, Decision Tree obtained better results although they make not much a difference. However, it is interesting to note that precision in Naive Bayes grows higher than in the first classifier. Also, common to both classifiers, the results suffer barely any change whether the dimensionality reduction is either of 50%, 25%, 10% or 5%. 

Finally, it should be pointed out that the training for Naive Bayes took longer than the training for Decision Tree Classifier.


