(ns assignment.scicloj
  (:require
    [aerial.hanami.templates :as ht]
    [assignment.eda :refer [liver-disease]]
    [calc-metric.patch]
    [nextjournal.clerk :as clerk]
    [scicloj.clay.v2.api :as clay]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [scicloj.ml.metamorph :as mm]
    [scicloj.noj.v1.vis.hanami :as hanami]
    [tech.v3.datatype.functional :as dfn]
    [utils.helpful-extracts :refer :all]))

;; ## Clojure with Smile Algorithm

;; Run clerk functions in comment to evaluate namespace in browser interactively
(comment
  (clerk/serve! {:browse? true :watch-paths ["."]})
  (clerk/show! "src/assignment/linear_regression.clj")
  (clay/start!))

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

;; ### Extract models
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


(def models-lasso-val-2
  (->> (best-models eval-lasso-val)
      (sort-by :other-metric-2)))

(-> models-lasso-val-2 first :summary)
(-> models-lasso-val-2 first :params)


;; ### Build final models for evaluation
;; #### Ridge
(def eval-ridge
  (evaluate-pipe
    (->> (extract-params models-ridge-val 3)
         (map ridge-pipe-fn))
    ds-split))

(def models-ridge
  (->> (best-models eval-ridge)
       reverse))

;; #### Lasso
(def eval-lasso
  (evaluate-pipe
    (->> (extract-params models-lasso-val-2 3)
         (map lasso-pipe-fn))
    ds-split))

(def models-lasso
  (->> (best-models eval-lasso)
       reverse))

;;#### Elastic net
(def eval-enet
  (evaluate-pipe
    (->> (extract-params models-enet-val 5)
         (map elastic-net-pipe-fn))
    ds-split))

(def models-enet
  (-> (best-models eval-enet)
      reverse))

;; ### Evaluate
(def ds-ridge
  (-> (model->ds models-ridge 3)
      (ds/rename-columns {:lambda :lambda2})
      (ds/add-columns {:lambda1 0 :alpha 0})))

(def ds-lasso
  (-> (model->ds models-lasso 3)
      (ds/rename-columns {:lambda :lambda1})
      (ds/add-columns {:lambda2 0 :alpha 1})))

(def ds-elastic
  (-> (model->ds models-enet 5)
      (ds/add-columns {:alpha (fn [ds]
                                (dfn// (:lambda1 ds) (dfn/+ (:lambda1 ds) (:lambda2 ds))))})))

(def col-order
  [:model-type :compute-time-ns :alpha :lambda1 :lambda2 :adj-r2 :mae :rmse])

;; ### Final comparisons
;; #### Ordered by alphas
(-> (model->ds (concat (ds/rows ds-ridge :as-maps) (ds/rows ds-lasso :as-maps) (ds/rows ds-elastic :as-maps)))
    (ds/reorder-columns col-order)
    (ds/order-by :alpha))

;; #### Ordered by Adjusted R$^2$
(-> (model->ds (concat (ds/rows ds-ridge :as-maps) (ds/rows ds-lasso :as-maps) (ds/rows ds-elastic :as-maps)))
    (ds/reorder-columns col-order)
    (ds/order-by :adj-r2 :desc))

((-> models-ridge first :pipe-fn)
 (merge (-> models-ridge first :fit-ctx)
        {:metamorph/data (ds/tail (:test (second ds-split)))
         :metamorph/mode :transform}))

;; ### Plots of ridge and lasso coefficients vs lambdas
(-> (coefs-vs-lambda liver-disease ridge-pipe-fn)
    (hanami/plot ht/line-chart
                 {:X     "log-lambda" :XSCALE {:zero false}
                  :Y     "coefficient" :YSCALE {:zero false}
                  :COLOR "predictor" :TITLE "Ridge"}))

(-> (coefs-vs-lambda liver-disease lasso-pipe-fn)
    (hanami/plot ht/line-chart
                 {:X     "log-lambda" :XSCALE {:zero false}
                  :Y     "coefficient" :YSCALE {:zero false}
                  :COLOR "predictor" :TITLE "Lasso"}))

; Note how regressor coefficients for lasso models may go to 0, whereas in a ridge model, regressor coefficients will only ever *approach* 0 while never reaching.