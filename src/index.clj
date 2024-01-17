(ns index)

;# Assignment 1: Linear Regression
;
;Instructions:
;
;* Select a suitable dataset from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php)
;* Scale, normalize, and/or encode your data appropriately.
;* Implement the Multivariate Linear Regression.
;* Compare the performance of your implementation in terms of computation time and other appropriate metrics using 5-fold Cross Validation using a variety of Alpha values.
;* Modify the "README.md" file to include the following sections:<br>
;&nbsp;&nbsp;&nbsp;&nbsp;* **Summary**: A one-paragraph summary of the algorithm that was implemented including any pertinent or useful information. If mathematics are appropriate, include those as well.<br>
;&nbsp;&nbsp;&nbsp;&nbsp;* **Results**: Display the comparative analysis of your implementation and the benchmark implementation in terms of computation time and other metrics. Which algorithm performed better? Why do you think this is the case?<br>
;&nbsp;&nbsp;&nbsp;&nbsp;* **Reflection**: One paragraph describing useful takeaways from the week, things that surprised you, or things that caused you inordinate difficulty.
;* Make sure that your README file is formatted properly and is visually appealing. It should be free of grammatical errors, punctuation errors, capitalization issues, etc.
;
;What I did:
;
;* Selected [Liver Disorders](https://archive.ics.uci.edu/dataset/60/liver+disorders). Interesting dataset that has been [misunderstood](https://www-sciencedirect-com.ezproxy.bgsu.edu/science/article/pii/S0167865516000088) in most publications.
;* Scaled all regressors and removed an indicator column called `:selector`.
;* Implemented three popular regularization techniques--ridge, lasso, and elastic net regression.
;* Compared Clojure's native Scicloj machine learning [library](https://github.com/scicloj/scicloj.ml) that leverages Java's Smile machine learning [library](https://haifengl.github.io/) (no alpha parameter, so I used lambda1 and lambda2) against Clojure's implementation of Python's [Scikit Learn](https://scikit-learn.org/stable/) library (does have alpha values I configured) using sklearn-clj [library](https://github.com/scicloj/sklearn-clj).