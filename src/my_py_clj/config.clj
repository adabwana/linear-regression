(ns my-py-clj.config
  (:require [libpython-clj2.python :as py]))

;; When you use conda, it should look like this.
(py/initialize! :python-executable "/opt/anaconda3/envs/my_env/bin/python3.7"
                :library-path "/opt/anaconda3/envs/my_env/lib/libpython3.7m.dylib")
