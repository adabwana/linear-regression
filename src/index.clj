(ns index)

;; # CS7200_SP2024_A01_Salvo
;
;The descriptions below reference linear_regression.html found in the LinearRegression folder. The subfolder linear_regression_files is required to see the HTML nicely formatted. So, you can download the LinearRegression folder and open the HTML inside.
;## Summary
;
;This assignment implemented three popular regularization techniques in regression: ridge, lasso, and elastic net, on a [liver disorder dataset](https://archive.ics.uci.edu/dataset/60/liver+disorders) found on UCI's machine learning repository. The purpose of these regularization techniques is to prevent overfitting in machine learning models, especially in cases with many features. They work by adding a penalty term to the loss function of the model, discouraging the model from assigning too much importance to any one feature. Mathematically, the penalty terms can be written as:
;
;### Ridge
;$$\text{Ridge Penalty} = \text{Loss Function} + \lambda \sum_{i=1}^{p} (\beta_i)^2$$
;
;Ridge regression adds a penalty equivalent to the square of the coefficients. The penalty term is controlled by a tuning parameter, lambda ($λ$). Ridge regression is also called "L2 regularization technique." Because of this L2, we may see a ridge regression lambda written as $\lambda_2$.
;
;In this technique, no feature is eliminated. Rather, coefficients "shrink" towards zero.
;
;### Lasso
;```math
;\text{Lasso Penalty} = \text{Loss Function} + \lambda \sum_{i=1}^{p} |\beta_i|
;```
;
;Lasso regression also adds a penalty term to the loss function. However, here, it's the absolute value of the  coefficients. Lasso regression is also called "L1 regularization technique." Because of this L1, we may see a lasso regression lambda written as $\lambda_1$.
;
;This technique allows for poorly performing features to be eliminated from the model.
;
;### Elastic Net
;$$\text{Elastic Net Penalty} = \text{Loss Function} + \lambda_1 \sum_{i=1}^{p} |\beta_i| + \lambda_2 \sum_{i=1}^{p} (\beta_i)^2$$
;
;Elastic Net is a combination of Ridge and Lasso regressions. It includes both L1 and L2 penalties. No feature is eliminated.
;
;#### Note on $\text{Loss Function}$
;In most regression models I've work with, the $\text{Loss Function}$ is the $\text{Residual Sum of Squares}$ ($RSS$).
;
;```math
;\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
;```
;
;## Results
;
;Below is a section of a table produced in my linear_regression.html.
;
;|                   :model-type | :compute-time-ns |     :alpha |   :lambda1 |   :lambda2 |    :adj-r2 |       :mae |      :rmse |
;|-------------------------------|-----------------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
;| :smile.regression/elastic-net |          2082857 | 0.93939394 | 0.93939394 | 0.06060606 | 0.25268445 | 2.13123998 | 2.83213976 |
;| :smile.regression/elastic-net |          1908943 | 0.94642857 | 0.53535354 | 0.03030303 | 0.25258695 | 2.13083904 | 2.83204344 |
;|       :smile.regression/ridge |          2079280 | 0.00000000 | 0.00000000 | 0.04000000 | 0.25257129 | 2.13024455 | 2.83164653 |
;|       :smile.regression/lasso |          1261144 | 1.00000000 | 0.06000000 | 0.00000000 | 0.25256741 | 2.13022717 | 2.83164094 |
;|       :smile.regression/lasso |          1166866 | 1.00000000 | 0.04000000 | 0.00000000 | 0.25256323 | 2.13022855 | 2.83164707 |
;
;Keeping in mind that these models are the best of their respective regularization techniques from a selection of 30 models with variable $\lambda$s; most generally, we see that all models perform near identically, especially in terms of Adjusted R$^2$ and errors (MAE and RMSE).
;
;When we look at compute time in nanoseconds, we see an advantage in lasso regression. In this subset of the final table in my HTML, we see compute times between ridge and elastic net to be almost equal. On average, though, elastic net models took the longest.
;
;As to why elastic net is the slowest, I am not surprised compute time is strictly greater than its constituent parts. As to why lasso regression is faster than ridge regression, I am somewhat confounded. Deductively, we could conclude that computing a square is more computationally heavy than computing an absolute value. Why that is, I do not know.
;
;The analogy I use to think about ridge and lasso regression is using scissors to cut around a shape. Ridge is precision getting closer and closer to that line to cut around. Lasso is more haphazard cutting, slicing out a piece inside the line because the result is still shapely, perhaps less perfectly so. Which one is going to finish cutting out the shape first? Probably Miss Lasso.
;
;All in all, I enjoy a parsimonious model. As such, if I can use a variable selection technique, I'm for it. This already leads me into preferring lasso. The icing on the cake is the best compute time. Therefore, even though no features were excluded, I prefer lasso.
;
;## Reflection
;
;I like to believe I understand linear regression at an advanced level. This assignment reminds me how deep simple things can be.
;
;#### Some depth comes from function implementation.
;Many, maybe most elastic net functions will take two tuning parameters, alpha and lambda ratio. See Python's ElasticNet implementation [documentation](https://ibex.readthedocs.io/en/latest/api_ibex_sklearn_linear_model_elasticnet.html), for example. I'm migrating from R to Clojure. Many of Clojure's machine learning algorithms use Java's Smile library. Smile's implementation of elastic-net takes in a lambda1 and lambda2. See [documentation](https://haifengl.github.io/api/java/smile/regression/ElasticNet.html). It also doesn't allow you to have either lambda be 0. If you want $lambda1 = 0$, Clojure throws an errors saying, in effect, "Why don't you do a Ridge Regression?" So instead of bundling everything in an elastic net, I had to separate out the three.
;
;The next question is; How does lambda1 and lambda2 relate to alpha? I couldn't find a direct answer, so instead, I used a hacky ratio found near the bottom of page 4 in [this paper](https://hastie.su.domains/Papers/elasticnet.pdf). Is this hacky alpha the same alpha Python uses? NO! Nevertheless, he persisted.
;
;#### Some depth comes from needing customization.
;In this assignment, we needed to measure computation time. In Clojure, there is a function called `time`. When evaluating a pipeline, there is an option that allows us to use a function as an evaluation metric. Great. Let me add the `time` function to :other-metrics. It didn't work. "A macro(!) can't be used as an evaluation metric." Who knew?
;
;Lo and behold, deep in the [docs](https://github.com/scicloj/metamorph.ml/blob/main/src/scicloj/metamorph/ml.clj#L98) and deep in the object structure was collected compute time. Great. I call it and see values of 0 to 2 per model. I discovered that `(System/currentTimeMillis)` returns integers only. Not granular enough. Looking at the function in the doc, notice it's a private function. How do I update that tiny private function in a huge namespace? I'm not going to fork it for a single use case. Turns out I had to write a patch file. That was new for me.
;
;#### Some depth comes from getting the language to do what I want it to do.
;Last, the section "Extract evaluation metrics from the best models" took me a lot of time. Before writing functions, I tried defining variables and reworking the variable to get it into the structure I wanted it. In Clojure, datasets are maps such that the keys//value pair correspond to column name//vector of values. So the table below is represented as {:a [1 2] :b [3 4] :c [5 6]}.
;
;|:a | :b | :c |
;|---|---:|--:|
;| 1 | 3 | 5 |
;| 2 | 4 | 6 |
;
;It's not very easy to add a row, for example. You have to match keys and conjoining vectors. Easier said than done. I think of these data structures having mobility maps, think the Vitruvian Man by Leonardo da Vinci. Man can touch things above, below, behind, in front, all kinds of directions. Think about a dog. Put a treat between its shoulders. It can't reach with its paws, its head can't turn that far; its mobility is limited. When I was working with these datasets, I felt similarly. With time comes comfort.
;
;#### Overall.
;Overall, I learned a lot and look forward to more.