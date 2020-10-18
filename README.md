# UL-Assignment-2

This assignment applies different dimensionality reduction techniques to two different types of datasets to see which D.R methods work best on which datasets. The D.R methods used are PCA, metric MDS and IsoMap. The datasets used are pixel data from a digital writting of a number (5 or 8) and pixel data of breast cancer images. For the former, we use D.R to cluster the correct digits together. For the later we aim to cluster the images with regards to malignant or benign.

The accuracy measure used for the D.R methods are KNN. KNN is computed on the original dataset and then again on the dimensionally reduced dataset and the difference between the two accuracies will determine the accuracy of the method.
