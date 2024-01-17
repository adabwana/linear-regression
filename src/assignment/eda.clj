(ns assignment.eda
  (:require
    [aerial.hanami.templates :as ht]
    [clojure.math.combinatorics :as combo]
    [fastmath.stats :as stats]
    [scicloj.kindly.v4.kind :as kind]
    [scicloj.ml.dataset :as ds]
    [scicloj.noj.v1.vis.hanami :as hanami]))

;; ## Exploratory Data Analysis
;; Load data
(defonce liver-disease
         (ds/dataset "data/bupa.csv"
                     {:key-fn (fn [colname]
                                (-> colname
                                    (clojure.string/replace #"\.|\s" "-")
                                    clojure.string/lower-case
                                    keyword))}))

;; ## Tables and nature of data
; First seven of data
(ds/head liver-disease 7)
; Descriptive statistics of columns
(ds/info liver-disease)

;; ### What is column :selector?
(set (:selector liver-disease))

; `:selector` takes two value only, 0 and 1. Let's see descriptive statistics of each group.

(ds/info (tech.v3.dataset/filter liver-disease #(= (:selector %) 1)))
(ds/info (tech.v3.dataset/filter liver-disease #(= (:selector %) 2)))

; The data looks similar in terms of summary statistics per column in either selector equals 1 or 2, above. Three columns I am focusing on are, mean, standard deviation, and skew. Mean, standard deviation, and skew are similar for both groups in `:mcv` and `:alkphos`. The other four columns deviate in some way from the other `:selector` groups in either mean, standard deviation, and/or skew.

;; ## Plots
(def cols-of-interest
  (remove #{:selector} (ds/column-names liver-disease)))

;; ### Linearity with response and histogram of regressor.
^kind/vega
(let [dataset liver-disease
      make-plot (fn [field]
                  (-> dataset
                      (hanami/vconcat {}                    ;can switch to hconcat
                                      [(hanami/plot dataset ht/point-chart
                                                    {:X field :XSCALE {:zero false}
                                                     :Y :drinks :YSCALE {:zero false}})
                                       (hanami/histogram dataset field {:nbins 20})])))]
  (->> (map make-plot cols-of-interest) (hanami/hconcat {} {})))

(comment                                                    ;make plot alternative were I can change height and width but not nbins
  (let [dataset liver-disease]
    (-> dataset
        (hanami/hconcat {}
                        [(hanami/plot dataset ht/point-chart
                                      {:HEIGHT 125 :WIDTH 175
                                       :X      :mcv :XSCALE {:zero false}
                                       :Y      :drinks :YSCALE {:zero false}})
                         (hanami/plot dataset ht/bar-chart
                                      {:HEIGHT 125 :WIDTH 175
                                       :X      field :YAGG :count})]))))

;; ### Pairs-plots.
^kind/vega
(let [data (ds/rows liver-disease :as-maps)
      column-names cols-of-interest]
  {:data   {:values data}
   :repeat {:column column-names
            :row    column-names}
   :spec   {:height   100 :width 100
            :mark     "circle"
            :encoding {:x {:field {:repeat "column"} :type "quantitative" :scale {:zero false}}
                       :y {:field {:repeat "row"} :type "quantitative" :scale {:zero false}}}}})

(let [combos (combo/combinations cols-of-interest 2)]
  (for [[x y] combos]
    (assoc {} [x y] (stats/correlation (get liver-disease x) (get liver-disease y)))))

; We can see pearson correlation between each pair of variables of interest. Correlations closer to |1| represent variables that have strong relationship to each other. For example, Looking at both the pairs-plot and the correlations, we see `:sgpt` `:sgot` are the most related variables and are related in the positive direction. Below we focus on the correlation of regressors on the response *only*.

(let [combos (combo/combinations cols-of-interest 2)]
  (for [[x y] combos
        :when (or (= :drinks x) (= :drinks y))]
    (assoc {} [x y] (stats/correlation (get liver-disease x) (get liver-disease y)))))

(comment                                                    ;another way to make histograms for diagonals
  (let [data (ds/rows liver-disease :as-maps)
        column-names cols-of-interest]
    {:data   {:values data}
     :repeat {:column column-names}
     :spec   {:mark     "bar"
              :encoding {:x {:field {:repeat "column"} :type "quantitative"}
                         :y {:aggregate "count"}}}})

  (stats/correlation-matrix (ds/columns liver-disease) :spearman))

;; ### Box-plots.
^kind/vega
(let [data (ds/rows liver-disease :as-maps)
      column-names (remove #{:selector} (ds/column-names liver-disease))]
  {:data   {:values data}
   :repeat {:column column-names}
   :spec   {:width    60 :mark "boxplot"
            :encoding {:y {:field {:repeat "column"} :type "quantitative" :scale {:zero false}}}}})

;; Looking at the box-plots, the circle points are outliers, viz. points outside Q1 - 1.5*IQR or Q3 + 1.5*IQR. Below we count the number of outliers per column.

(let [columns cols-of-interest]
  (for [column columns]
    (assoc {} column (count (stats/outliers (get liver-disease column))))))

(comment
  ; created permutation of columns of interest. Wanted combinations, below
  (for [x cols-of-interest
        y (rest cols-of-interest)
        :when (not= x y)]
    [x y]))