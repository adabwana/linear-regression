(ns utils.eda
  (:require
    [aerial.hanami.templates :as ht]
    [fastmath.stats :as stats]
    [scicloj.ml.dataset :as ds]
    [scicloj.noj.v1.stats :as vis-stats]
    [scicloj.noj.v1.vis.hanami :as hanami]))


(defn keys-in
  "Returns a sequence of all key paths in a given map using DFS walk." ; Function description
  [m]                                                       ; m: the input map
  (letfn [(children [node]                                  ; Define a function to get the children of a node
            (let [v (get-in m node)]                        ; Get the value at the node's path in the map
              (if (map? v)                                  ; If the value is a map
                (map (fn [x] (conj node x)) (keys v))       ; Recursively call children on the value
                [])))                                       ; Return an empty list if the value is not a map
          (branch? [node] (-> (children node) seq boolean))] ; Define a function to check if a node is a branch
    (->> (keys m)                                           ; Thread-last macro to process the keys of the input map
         (map vector)                                       ; Map the keys to vectors
         (mapcat #(tree-seq branch? children %)))))         ; Use mapcat and tree-seq to perform a depth-first search

(defn path-to
  "Returns system path of the filename being called."
  [filename]
  (let [default-path (System/getProperty "user.dir")        ; Get the default directory
        data-path (str default-path "/data/")               ; Create the data directory path
        full-path (str data-path filename)]                 ; Create the full file path
    full-path))                                             ; Return the full file path

(defn augment
  "Applies a series of transformations to the input data and returns the augmented dataset."
  [data y x & [args]]
  (let [predicted (-> data
                      (vis-stats/add-predictions y [x args]
                                                 {:model-type :smile.regression/ordinary-least-square}))
        residuals (map - ((keyword y) predicted) ((keyword (subs (str y "-prediction") 1)) predicted))
        std-resid (stats/standardize residuals)             ; or stats/robust-standardize
        sqrt-abs-std-resid (->> std-resid
                                (map abs)
                                (map #(Math/pow % 0.5)))]
    (-> predicted
        (ds/add-column :resid residuals)
        (ds/add-column :std-resid std-resid)
        (ds/add-column :sqrt-abs-std-resid sqrt-abs-std-resid))))

(defn all-combinations [coll]
  (letfn [(comb [coll]
            (if (empty? coll)
              [[]]
              (let [rest (comb (rest coll))]
                (concat rest (map #(cons (first coll) %) rest)))))]
    (rest (comb coll))))                                    ; rest removes the empty set

;; ### 1. Check for linearity between different variables ###
(defn linearity-plot
  "Generates a linearity plot for the given data, response variable, and predictor variable."
  [data y x & [args]]
  (let [df (augment data y x args)]
    (-> df
        (hanami/layers {}                                   ; or nil
                       [(hanami/plot {} ht/point-chart
                                     {:TITLE "Residual Plot"
                                      :X     x :XSCALE {:zero false}
                                      :Y     :resid :YSCALE {:zero false}})
                        (hanami/plot {} ht/line-chart
                                     {:X         x
                                      :Y         :resid
                                      :TRANSFORM [{:loess :Y :on :X}]
                                      :MSIZE     3 :MCOLOR "gray" :MSDASH [8 4]})
                        {:mark     {:type "rule" :tooltip true}
                         :encoding {:y          {:datum 0}
                                    :strokeDash {:value [4 4]}
                                    :color      {:value "firebrick"}}}]))))

;; ### 2. Check for normality of random error ###
(defn resid-normality-plot [data y x & [args]]
  (let [df (augment data y x args)]
    (-> df
        (hanami/layers {}                                   ; or nil
                       [(hanami/plot {} ht/point-chart
                                     {:TITLE     "QQ-Plot"
                                      :TRANSFORM [{:quantile :std-resid :step 0.01 :as ["prob" "value"]}
                                                  {:calculate "quantileNormal(datum.prob)" :as "norm"}]
                                      :ENCODING  {:x {:field "norm" :type "quantitative" :zero false}
                                                  :y {:field "value" :type "quantitative" :zero false}}})
                        {:data     {:values [{:xx -3 :yy -3} {:xx 3 :yy 3}]}
                         :mark     "line"
                         :encoding {:x          {:field :xx :type "quantitative"}
                                    :y          {:field :yy :type "quantitative"}
                                    :strokeDash {:value [4 4]}
                                    :color      {:value "firebrick"}}}]))))

;; ### 3. Check for zero mean and constant variance of random error ###
(defn scale-location-plot [data y x & [args]]
  (let [df (augment data y x args)]
    (-> df
        (hanami/layers {}                                   ; or nil
                       [(hanami/plot {} ht/point-chart
                                     {:TITLE  "Scale-location"
                                      :X      (keyword (subs (str y "-prediction") 1))
                                      :Y      :sqrt-abs-std-resid
                                      :XSCALE {:zero false} :YSCALE {:zero false}
                                      :XAXIS  "y-prediciton" :YTITLE "sqrt-abs-std-res"})
                        (hanami/plot {} ht/line-chart
                                     {:X         (keyword (subs (str y "-prediction") 1))
                                      :Y         :sqrt-abs-std-resid
                                      :TRANSFORM [{:loess :Y :on :X}]
                                      :MSIZE     3 :MCOLOR "gray" :MSDASH [8 4]})
                        {:mark     {:type "rule" :tooltip true}
                         :encoding {:y          {:field :sqrt-abs-std-resid :aggregate "mean"}
                                    :strokeDash {:value [4 4]}
                                    :color      {:value "firebrick"}}}]))))

;; ### 4. Check for independence of random error ###
(defn resid-independence-plot [data y x & [args]]
  (let [df (augment data y x args)]
    (-> (ds/dataset {:x (range 0 (ds/row-count df) 1)
                     :y (:resid (ds/order-by df y))})
        (hanami/layers {}                                   ; or nil
                       [(hanami/plot {} ht/point-chart
                                     {:TITLE "Check for Independence Residuals sorted by y"
                                      :X     :x :Y :y
                                      :XAXIS "Row Number" :YTITLE "Residuals"})
                        (hanami/plot {} ht/line-chart
                                     {:X         :x
                                      :Y         :y
                                      :TRANSFORM [{:loess :Y :on :X}]
                                      :MSIZE     3 :MCOLOR "gray" :MSDASH [8 4]})
                        {:mark     {:type "rule" :tooltip true}
                         :encoding {:y          {:datum 0}
                                    :strokeDash {:value [4 4]}
                                    :color      {:value "firebrick"}}}]))))

;; ### Final diagnostic plot often presented, too
(defn generate-data [df y]
  (let [y-keyword (keyword y)
        y-pred-keyword (keyword (subs (str y "-prediction") 1))]
    [{:xx (min (Math/round (apply min (y-keyword df)))
               (Math/round (apply min (y-pred-keyword df))))
      :yy (min (Math/round (apply min (y-keyword df)))
               (Math/round (apply min (y-pred-keyword df))))}
     {:xx (min (Math/round (apply max (y-keyword df)))
               (Math/round (apply max (y-pred-keyword df))))
      :yy (min (Math/round (apply max (y-keyword df)))
               (Math/round (apply max (y-pred-keyword df))))}]))

(defn obs-pred-plot [data y x & [args]]
  (let [df (augment data y x args)]
    (-> df
        (hanami/layers {}                                   ; or nil
                       [(hanami/plot {} ht/point-chart
                                     {:TITLE  "Observed vs Predicted Values"
                                      :X      (keyword (subs (str y "-prediction") 1))
                                      :Y      y :XAXIS "Predicted Values" :YAXIS "Actual Values"
                                      :XSCALE {:zero false} :YSCALE {:zero false}})
                        (hanami/plot {} ht/line-chart
                                     {:X     (keyword (subs (str y "-prediction") 1))
                                      :Y     y :TRANSFORM [{:loess :Y :on :X}]
                                      :MSIZE 3 :MCOLOR "gray" :MSDASH [8 4]})
                        {:data     {:values (generate-data df y)}
                         :mark     {:type "line" :tooltip true}
                         :encoding {:x          {:field :xx :type "quantitative"}
                                    :y          {:field :yy :type "quantitative"}
                                    :strokeDash {:value [4 4]}
                                    :color      {:value "firebrick"}}}]))))

