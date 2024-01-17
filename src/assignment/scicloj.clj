(ns assignment.scicloj
  (:require
    [assignment.eda :refer [liver-disease]]
    [calc-metric.patch] ;eval from milliseconds to nanoseconds
    [nextjournal.clerk :as clerk]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [scicloj.ml.metamorph :as mm]
    [tech.v3.datatype.functional :as dfn]
    [utils.helpful-extracts
     :refer [best-models evaluate-pipe extract-params model->ds]]))

;; ## Clojure with Smile Algorithm
;; Define regressors and response
(def response :drinks)
(def regressors
  (remove #{response} (ds/column-names liver-disease)))

;; ## Build pipelines
(def pipeline-fn
  (ml/pipeline
    (mm/remove-column :selector)
    (mm/std-scale regressors {})
    (mm/set-inference-target response)))

; I'm building three different pipelines because smile's implementation of elastic net, the parameters are lambda1 and lambda2. I cannot make either of those parameters 0. If I do, the model throws and errors and says to try the explicit model. That is, if $lambda1 = 0$, elastic net throws an error saying, "try using a ridge regression model."

(defn ridge-pipe-fn [params]                                ;alpha = 0
  (ml/pipeline
    pipeline-fn
    {:metamorph/id :model}
    (mm/model (merge {:model-type :smile.regression/ridge}
                     params))))

(defn lasso-pipe-fn [params]                                ;alpha = 1
  (ml/pipeline
    pipeline-fn
    {:metamorph/id :model}
    (mm/model (merge {:model-type :smile.regression/lasso}
                     params))))

(defn elastic-net-pipe-fn [params]                          ;0 < alpha < 1
  (ml/pipeline
    pipeline-fn
    {:metamorph/id :model}
    (mm/model (merge {:model-type :smile.regression/elastic-net}
                     params))))

;; ## Partition data
(def ds-split
  (ds/split->seq
    liver-disease
    :kfold {:seed 123 :k 5 :ratio [0.8 0.2]}))              ;:split-names [:train-val :test]

(def train-val-splits
  (ds/split->seq
    (:train (first ds-split))
    :kfold {:seed 123 :k 5}))

;; ## Build models
;; ### Ridge
(def ridge-pipelines
  (->> (ml/sobol-gridsearch {:lambda (ml/linear 0 1000 250)})
       (map ridge-pipe-fn)))

(def eval-ridge-val
  (evaluate-pipe ridge-pipelines train-val-splits))

;; ### Lasso
(def lasso-pipelines
  (->> (ml/sobol-gridsearch {:lambda (ml/linear 0 1000 250)})
       (map lasso-pipe-fn)))

(def eval-lasso-val
  (evaluate-pipe lasso-pipelines train-val-splits))

;; ### Elastic Net
(def elastic-pipelines
  (->> (ml/sobol-gridsearch {:lambda1 (ml/linear 0.0001 100 16)
                             :lambda2 (ml/linear 0.0001 100 16)})
       (map elastic-net-pipe-fn)))

(comment
  (def elastic-pipelines
    (->> (ml/sobol-gridsearch
           (dissoc
             (ml/hyperparameters :smile.regression/elastic-net)
             :tolerance :max-iterations))
         (take 500)
         (map elastic-net-pipe-fn))))

(def eval-enet-val
  (evaluate-pipe elastic-pipelines train-val-splits))

;; ## Extract models
(def models-ridge-val
  (-> (best-models eval-ridge-val)
      reverse))

(def models-lasso-val
  (-> (best-models eval-lasso-val)
      reverse))

(def models-enet-val
  (-> (best-models eval-enet-val)
      reverse))

;; ### Best model for each pipeline
^{::clerk/viewer clerk/code}
(-> models-ridge-val first :summary)
^{::clerk/viewer clerk/code}
(-> models-lasso-val first :summary)
^{::clerk/viewer clerk/code}
(-> models-enet-val first :summary)

; The summary for our best lasso model looks wrong. A negative Adjusted R$2$? We can see an issue derives from not calculating the Adjusted R$^2$, i.e the `:metric`.

(-> models-lasso-val first :metric)

; Instead, we will collect the best lasso models according to the lowest `mae`, i.e. `:other-metric-1`.

(def models-lasso-val-2
  (->> (best-models eval-lasso-val)
       (sort-by :other-metric-1)))

(-> models-lasso-val-2 first :summary)

;; ## Build final models for evaluation
;; ### Ridge
(def eval-ridge
  (evaluate-pipe
    (->> (extract-params models-ridge-val 3)                ;use best 3 lambdas
         (map ridge-pipe-fn))
    ds-split))

(def models-ridge
  (->> (best-models eval-ridge)
       reverse))

;; ### Lasso
(def eval-lasso
  (evaluate-pipe
    (->> (extract-params models-lasso-val-2 3)              ;use best 3 lambdas
         (map lasso-pipe-fn))
    ds-split))

(def models-lasso
  (->> (best-models eval-lasso)
       reverse))

;; ### Elastic net
(def eval-enet
  (evaluate-pipe
    (->> (extract-params models-enet-val 3)                 ;use best 3 lambda1s and lambda2s
         (map elastic-net-pipe-fn))
    ds-split))

(def models-enet
  (-> (best-models eval-enet)
      reverse))

;; ## Build final models for evaluation
(def ds-ridge
  (-> (model->ds models-ridge 3)
      (ds/rename-columns {:lambda :lambda2})
      (ds/add-columns {:lambda1 0 :alpha 0})))

(def ds-lasso
  (-> (model->ds models-lasso 3)
      (ds/rename-columns {:lambda :lambda1})
      (ds/add-columns {:lambda2 0 :alpha 1})))

(def ds-elastic
  (-> (model->ds models-enet 3)
      (ds/add-columns {:alpha (fn [ds]
                                (dfn// (:lambda1 ds) (dfn/+ (:lambda1 ds) (:lambda2 ds))))})))

(def col-order
  [:model-type :compute-time-ns :alpha :lambda1 :lambda2 :adj-r2 :mae :rmse])

;; ## Final comparisons
(def top-scicloj
  (-> (model->ds (concat (ds/rows ds-ridge :as-maps) (ds/rows ds-lasso :as-maps) (ds/rows ds-elastic :as-maps)))
      (ds/reorder-columns col-order)
      (ds/order-by :adj-r2 :desc)))

top-scicloj

(comment
  ((-> models-ridge first :pipe-fn)
   (merge (-> models-ridge first :fit-ctx)
          {:metamorph/data (ds/tail (:test (second ds-split)))
           :metamorph/mode :transform})))