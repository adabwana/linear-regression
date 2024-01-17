(ns assignment.sklearn
  (:require
    [assignment.eda :refer [liver-disease]]
    [assignment.scicloj :refer [col-order]]
    [calc-metric.patch] ;eval from milliseconds to nanoseconds
    [fastmath.stats :as stats]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [scicloj.ml.metamorph :as mm]
    [scicloj.sklearn-clj.ml]
    [utils.helpful-extracts
     :refer [best-models evaluate-pipe extract-params model->ds]]))

;; # Clojure with Scikit Learn Algorithm
; Define regressor and response
(def response :drinks)
(def regressors
  (remove ##(= response %) (ds/column-names liver-disease)))

;; ## Build pipelines
(def pipeline-fn
  (ml/pipeline
    (mm/remove-column :selector)
    (mm/std-scale regressors {})
    (mm/set-inference-target response)))

; In scikit.learn's implementation of elastic net, it takes an alpha value, where $alpha = 0$ is a ridge regression model, $alpha = 1$ is a lasso regression model, and $0 < alpha < 1$ is strictly an elastic net model that combines the loss functions of both ridge and lasso regression models at differing strengths. Closer to 0 means ridge regression loss function has a heavier consideration and alpha closer to 1 meaning the lasso loss function has a heavier consideration. (This is in general, there is an optional l1-ratio parameter that will change the above interpretation.)

(defn sklearn-pipe-fn [params]
  (ml/pipeline
    pipeline-fn
    {:metamorph/id :model}
    (mm/model (merge {:model-type     :sklearn.regression/elastic-net
                      :predict-proba? false}
                     params))))

;; ## Partition data
(def ds-split                                               ;:split-names [:train-val :test]
  (ds/split->seq liver-disease :kfold {:seed 123 :k 5 :ratio [0.8 0.2]}))

(def train-val-splits
  (ds/split->seq
    (:train (first ds-split))
    :kfold {:seed 123 :k 5}))

;; ## Evaluate pipelines
(def sklearn-pipelines
  (->>
    (ml/sobol-gridsearch {:alpha (ml/linear 0 1 250)})      ;doesnt like l1-ratio, why??
    (map sklearn-pipe-fn)))

(def evaluations-sklearn
  (ml/evaluate-pipelines
    sklearn-pipelines
    train-val-splits
    stats/omega-sq
    :accuracy
    {:other-metrices                   [{:name :mae :metric-fn ml/mae}
                                        {:name :rmse :metric-fn ml/rmse}]
     :return-best-pipeline-only        false
     :return-best-crossvalidation-only true}))

;; ## Extract models
(def models-sklearn-vals
  (->> (best-models evaluations-sklearn)
       reverse))

(-> models-sklearn-vals first :metric)
(-> models-sklearn-vals first :params)
(-> models-sklearn-vals first :fit-ctx :model :model-data :attributes :intercept_)
(-> models-sklearn-vals first :fit-ctx :model :model-data :attributes :coef_)

(-> (model->ds models-sklearn-vals 5)
    (ds/reorder-columns col-order))

(-> (model->ds models-sklearn-vals 5)
    (ds/reorder-columns col-order)
    (ds/order-by :adj-r2 :desc))

;; ## Build final models for evaluation
(def eval-sklearn
  (evaluate-pipe
    (->> (extract-params models-sklearn-vals 5)             ;use best 3 alphas
         (map sklearn-pipe-fn))
    ds-split))

(def models-sklearn
  (->> (best-models eval-sklearn)
       reverse))

(def top-sklearn
  (-> (model->ds models-sklearn 5)
      (ds/reorder-columns col-order)
      (ds/order-by :adj-r2 :desc)
      (ds/drop-columns :predict-proba?)))

top-sklearn