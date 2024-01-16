# This is a test
I want to deploy to https://adabwana.github.io/linear-regression. I tried copying the folder structure of kiramclean's [Clojure Tidy Tuesday](https://github.com/kiramclean/clojure-tidy-tuesdays). The only thing I saw different was in the build job. There were two things:

1) [My build](https://github.com/adabwana/linear-regression/actions/runs/7535518196/job/20511615004) pulls and builds Jekyll, and
2) In Upload artifact's first few lines of `Run actions/upload-pages-artifact@v3`, my `with` is in path: `./_site`.

[kiramclean's build](https://github.com/kiramclean/clojure-tidy-tuesdays/actions/runs/7507950544/job/20442529897) doesn't pull or build Jekyll. Her Upload artifact's `Run actions/upload-pages-artifact@v3` has path as `.`