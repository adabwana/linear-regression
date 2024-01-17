(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(defn build []
  (clay/make!
    {:format           [:quarto :html]
     :book             {:title "Linear Regression"}
     :base-source-path "src"
     :base-target-path "docs"
     :subdirs-to-sync  ["notebooks" "data"]
     :source-path      ["index.md" ;index.clj
                        "assignment/eda.clj"
                        "assignment/scicloj.clj"
                        "assignment/sklearn.clj"
                        "assignment/conclusion.clj"]}))

(comment
  (build))