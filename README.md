# CS7200_SP2024_A01_Salvo

The code in the `src/assignments` folder were built with [Clay](https://scicloj.github.io/clay/) and deployed with [Github Pages](https://pages.github.com/). Please review submission[here](https://adabwana.github.io/linear-regression/).

## Summary

In this assignment, I implemented three popular regularization techniques in regression: ridge, lasso, and elastic net, on a [liver disorder dataset](https://archive.ics.uci.edu/dataset/60/liver+disorders) found on UCI's machine learning repository. The purpose of these regularization techniques is to prevent overfitting in machine learning models, especially in cases with many features. They work by adding a penalty term to the loss function of the model, discouraging the model from assigning too much importance to any one feature. Mathematically, the penalty terms can be written as:

### Ridge
$$\text{Ridge Penalty} = \text{Loss Function} + \lambda \sum_{i=1}^{p} (\beta_i)^2$$

Ridge regression adds a penalty equivalent to the square of the coefficients. The penalty term is controlled by a tuning parameter, lambda ($λ$). Ridge regression is also called "L2 regularization technique." Because of this L2, we may see a ridge regression lambda written as $\lambda_2$.

In this technique, no feature is eliminated. Rather, coefficients "shrink" towards zero.

### Lasso
$$\text{Lasso Penalty} = \text{Loss Function} + \lambda \sum_{i=1}^{p} |\beta_i|$$

Lasso regression also adds a penalty term to the loss function. However, here, it's the absolute value of the coefficients. Lasso regression is also called "L1 regularization technique." Because of this L1, we may see a lasso regression lambda written as $\lambda_1$.

This technique allows for poorly performing features to be eliminated from the model.

### Elastic Net
$$\text{Elastic Net Penalty} = \text{Loss Function} + \lambda_1 \sum_{i=1}^{p} |\beta_i| + \lambda_2 \sum_{i=1}^{p} (\beta_i)^2$$

Elastic Net is a combination of Ridge and Lasso regressions. It includes both L1 and L2 penalties. Depending on the algorithm's implementation, features *may* or *may not* be eliminated.

For example, in Smile's implementation of elastic net, it can take $\lambda_1$ and $\lambda_2$. In this case, if $\lambda_2$ was set to zero, this is more strictly a lasso regression, thus allowing feature elimination. However, this point is mute in practice because Smile throws an error if either $\lambda_1$ or $\lambda_2$ is 0. 

In SciKit Learn's elastic net implementation, unless you explicitly set the [l1-ratio to 1](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) (meaning a L1 penalty, i.e. lasso regression), otherwise, the default is 0.5. In this case, no features will be eliminated regardless of the alpha, though the coefficients may get *veryy* close to 0. 

#### Note on $\text{Loss Function}$
In most regression models I've work with, the $\text{Loss Function}$ is the $\text{Residual Sum of Squares}$ ($RSS$).

$$\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

## Results

We were asked to test two implementations of the same algorithm. I ran Java's Smile in scicloj.clj and Python's Scikit Learn in sklearn.clj. Below are the two main tables produced in scicloj.clj and sklearn.clj:

|                   :model-type | :compute-time-ns |     :alpha |     :lambda1 |     :lambda2 |    :adj-r2 |       :mae |      :rmse |
|-------------------------------|-----------------:|-----------:|-------------:|-------------:|-----------:|-----------:|-----------:|
|       :smile.regression/ridge |           691179 | 0.00000000 |   0.00000000 | 196.78714859 | 0.27138367 | 2.19656820 | 2.88244828 |
|       :smile.regression/ridge |           644802 | 0.00000000 |   0.00000000 | 192.77108434 | 0.27131830 | 2.19552025 | 2.88102276 |
|       :smile.regression/ridge |           843926 | 0.00000000 |   0.00000000 | 188.75502008 | 0.27124938 | 2.19445469 | 2.87959384 |
|       :smile.regression/lasso |          1401088 | 1.00000000 |   4.01606426 |   0.00000000 | 0.26141892 | 2.73072125 | 3.40320579 |
| :smile.regression/elastic-net |          1581917 | 0.62499984 | 100.00000000 |  60.00004000 | 0.26059398 | 2.22496112 | 2.92246056 |
| :smile.regression/elastic-net |          1193860 | 0.65217371 | 100.00000000 |  53.33338000 | 0.26021738 | 2.22204401 | 2.91718016 |
|       :smile.regression/lasso |          1240533 | 1.00000000 |   8.03212851 |   0.00000000 | 0.25986847 | 2.73272023 | 3.40683967 |
| :smile.regression/elastic-net |          1592694 | 0.68181793 | 100.00000000 |  46.66672000 | 0.25984550 | 2.21890789 | 2.91171010 |
|       :smile.regression/lasso |          1372893 | 1.00000000 |  12.04819277 |   0.00000000 | 0.25806319 | 2.73472124 | 3.41060937 |

|                     :model-type | :compute-time-ns |     :alpha |    :adj-r2 |       :mae |      :rmse |
|---------------------------------|-----------------:|-----------:|-----------:|-----------:|-----------:|
| :sklearn.regression/elastic-net |          4081852 | 0.98393574 | 0.26161668 | 2.30607905 | 3.04582307 |
| :sklearn.regression/elastic-net |          4460882 | 0.98795181 | 0.26159480 | 2.30679512 | 3.04693277 |
| :sklearn.regression/elastic-net |          4107027 | 0.99196787 | 0.26157229 | 2.30750974 | 3.04804261 |
| :sklearn.regression/elastic-net |          3735449 | 0.99598394 | 0.26154913 | 2.30822291 | 3.04915259 |
| :sklearn.regression/elastic-net |          4909871 | 1.00000000 | 0.26152532 | 2.30893464 | 3.05026269 |

In terms of the goodness-of-fit measures, both implementations perform similarly. The main difference is between compute times. Scikit Learn's implementation takes over twice the time as Smile's.

Choosing a best model, I'd pick Smile's ridge regression with a lambda of 196.78714859. It has the benefit of fastest computational time and best Adjusted R$^2$. The model coefficients are as follows:

## Reflection

I like to believe I understand linear regression at mid-to-high level. This assignment reminds me how deep deceptively simple things can be.

### Some depth comes from function implementation.
Many, maybe most elastic net functions will take two tuning parameters, alpha and lambda ratio. See Python's ElasticNet implementation [documentation](https://ibex.readthedocs.io/en/latest/api_ibex_sklearn_linear_model_elasticnet.html), for example. I'm migrating from R to Clojure. Many of Clojure's machine learning algorithms use Java's Smile library. Smile's implementation of elastic-net takes in a lambda1 and lambda2. See [documentation](https://haifengl.github.io/api/java/smile/regression/ElasticNet.html). It also doesn't allow you to have either lambda be 0. If you want $lambda1 = 0$, Clojure throws an errors saying, in effect, "Why don't you do a Ridge Regression?" So instead of bundling everything in an elastic net, I had to separate out the three--ridge, lasso, and elastic net.

The next question is; How does lambda1 and lambda2 relate to alpha? I couldn't find a direct answer, so instead, I used a hacky ratio found near the bottom of page 4 in [this paper](https://hastie.su.domains/Papers/elasticnet.pdf). Is this hacky alpha the same alpha Python uses? NO! Nevertheless, he persisted.

### Some depth comes from needing customization.
In this assignment, we needed to measure computation time. In Clojure, there is a function called `time`. When evaluating a pipeline, there is an option that allows us to use a function as an evaluation metric. Great. Let me add the `time` function to :other-metrics. It didn't work. "A macro(!) can't be used as an evaluation metric." Who knew?

Lo and behold, deep in the [docs](https://github.com/scicloj/metamorph.ml/blob/main/src/scicloj/metamorph/ml.clj#L98) and deep in the object structure was collected compute time. Great. I call it and see values of 0 to 2 per model. I discovered that `(System/currentTimeMillis)` returns integers only. Not granular enough. Looking at the function in the doc, notice it's a private function. How do I update that tiny private function in a huge namespace? I'm not going to fork it for a single use case. Turns out I had to write a patch file. That was new for me.

### Some depth comes from getting the language to do what I want it to do.
Last, the section "Extract evaluation metrics from the best models" took me a lot of time. Before writing functions, I tried defining variables and reworking the variable to get it into the structure I wanted it. In Clojure, datasets are maps such that the keys//value pair correspond to column name//vector of values. So the table below is represented as {:a [1 2] :b [3 4] :c [5 6]}.

|:a | :b | :c |
|---|---:|--:|
| 1 | 3 | 5 |
| 2 | 4 | 6 |

It's not very easy to add a row, for example. You have to match keys and conjoining vectors. Easier said than done. I think of these data structures having mobility maps, think the Vitruvian Man by Leonardo da Vinci. Man can touch things above, below, behind, in front, all kinds of directions. Think about a dog. Put a treat between its shoulders. It can't reach with its paws, its head can't turn that far; its mobility is limited. When I was working with these datasets, I felt similarly. With time comes comfort.

### Some depth comes from doing things I've never done before.
Lastly, I worked hard to get the results published on Github Pages (hopefully, I can figure our deployment on GitLab). I set up my project such that I could use Github's default "build from branch" function. The process involves using the `build` function in `env/dev/src/dev.clj`. Clay will then make quarto files and quarto will render them into the nice "book" organized in a `docs` folder that I push to my branch `gh-pages` where Github Actions is told to use as deployment. 

I believe my setup is technically correct (the page is up, and I'm pretty happy with that), but I am very interested in this aspect of clean code, clean architecture, modularization, reproducible results, test driven development, deployment, etc. In these respects, I am not satisfied. For this reason, my final project for this course I hope to study a topic in Machine Learning Operations.
