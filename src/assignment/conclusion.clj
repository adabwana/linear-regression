(ns assignment.conclusion
  (:require
    [assignment.scicloj :refer [models-ridge top-scicloj]]
    [assignment.sklearn :refer [top-sklearn]]))

;; # Conclusion

; For this assignment, we had to create a linear regression model and compare it against another implementation of that model. For our comparisons, we needed computation time to build each model along with a few other regression goodness-of-fit measures, either of the model (AIC, BIC, Mallows C$_p$, Adjusted R$^2$) or of the errors (RMSE, MAE, MAD).<br/><br/>

;; ## ML Process
; The process involved hyperparameter tuning for elastic net models built using Scikit Learn and Smile algorithms. The Machine learning process involved:<br/><br/>

; 1) Partitioning the data into training, validation, and testing.<br/>
; 2) With the training and validation data, we tune the hyperparameters (lambdas in Smile and alpha in Scikit Learn).<br/>
; 3) Using the best hyperparameters, we build a final model with the training data as the combined training and validation tested on the testing data.<br/><br/>

; The results of this process are in the two tables below:

top-scicloj
top-sklearn

(double (/ (apply min (:compute-time-ns top-sklearn))
           (apply max (:compute-time-ns top-scicloj))))

;; ## Final Remarks
; In terms of the goodness-of-fit, both implementations perform similarly. The main difference is between compute times. Scikit Learn's implementation takes 1.5 to 2.5 times longer than Smile's (multiple runs).

; Choosing a best model, I'd pick Smile's ridge regression with a lambda of 196.78714859. It has the benefit of fastest computational time and best Adjusted R$^2$. The model coefficients are as follows:

(-> models-ridge first :summary)

; Overall, the model is not a good fit. The model accounts for only 27.138% of the variance (note how the above summary is the model being evaluated on the *training data*, the tables above shows the metrics on the test data).

; Looking at the model we can say, that given every other variable remaining constant, for every one unit increase in :mcv, :drinks increases by 0.518; for every one unit increase in alkphos, :drinks increases by 0.096; for every one unit increase in :sgpt, :drinks increases by 0.041; for every one unit increase in :sgot, :drinks increases by 0.316; for every one unit increase in :gammagt, :drinks increases by 0.378.

(comment
  (-> models-ridge first :summary .intercept)
  (seq (-> models-ridge first :summary .coefficients))
  (-> models-ridge first :summary .formula))
