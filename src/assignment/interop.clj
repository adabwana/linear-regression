(ns assignment.interop
  (:require
    [calc-metric.patch]                                     ; ns instead of ms]
    [fastmath.stats :as stats]
    [nextjournal.clerk :as clerk]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [scicloj.ml.metamorph :as mm]
    [scicloj.sklearn-clj.metamorph :as sklearn-mm]
    [scicloj.sklearn-clj.ml]
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python :refer [py. py.. py.-] :as py]))

(require-python '[numpy :as np])
(require-python '[pandas :as pd])
(require-python 'itertools
                '(itertools product))
(require-python 'sklearn.model_selection
                '(sklearn.model_selection train_test_split GridSearchCV))
(require-python 'sklearn.linear_model
                '(sklearn.linear_model ElasticNet))
(require-python 'sklearn.metrics
                '(sklearn.metrics mean_absolute_error mean_squared_error r2_score))
(require-python 'time
                '(time time))

;; Run clerk functions in comment to evaluate namespace in browser interactively
(comment
  (clerk/serve! {:browse? true :watch-paths ["."]})
  (clerk/show! "src/assignment/interop.clj"))

;; ### Load data
(defonce autompg
         (ds/dataset "data/bupa.csv"
                     {:key-fn (fn [colname]
                                (-> colname
                                    (clojure.string/replace #"\.|\s" "-")
                                    clojure.string/lower-case
                                    keyword))}))
