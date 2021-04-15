# From Philosophy to Interfaces: an Explanatory Method and a Tool Inspired by Achinstein’s Theory of Explanation

In this repository you'll find the source code and the data used in the experiments of our paper: "From Philosophy to Interfaces: an Explanatory Method and a Tool Inspired by Achinstein’s Theory of Explanation".
This paper has been presented at [ACM IUI 2021](https://iui.acm.org/2021/), the 26th annual meeting of the intelligent interfaces, in April the 15th.

The code of the baseline is at [software/yai_baseline](software/yai_baseline).
The code of our user-centric variation of the baseline is at [software/yai_alternative](software/yai_alternative).

The HELOC dataset and more information about it, including instructions to download, can be found at [https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=2](https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=2). Copy and paste the file heloc_dataset.csv into [software/yai_baseline/aix](software/yai_baseline/aix) and [software/yai_alternative/aix](software/yai_alternative/aix)

The collected user studies are in [user_study/data](user_study/data).
To generate the boxplots and compute the related statistics, run [user_study/result_analyser.py](user_study/result_analyser.py).