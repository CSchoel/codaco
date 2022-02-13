# Complete Data Count (Codaco)

[![build](https://github.com/CSchoel/codaco/actions/workflows/ci.yaml/badge.svg)](https://github.com/CSchoel/codaco/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/CSchoel/codaco/branch/main/graph/badge.svg?token=0092A8I7WF)](https://codecov.io/gh/CSchoel/codaco)

Codaco ~~is~~ will maybe one day be a data science library for exploring unknown datasets.
The name is inspired by the "complete blood count" used to screen for diseases.

My vision with this project is to tackle the "garbage in, garbage out" problem of data science.
I have seen both students and scientists work with data that is simply not worth their time, because it is biased, noisy, and overall unfit for the task at hand.
This observation does very much include myself, like that one time I spent an entire semester working on an ML project involving ECGs until I found out that all that my algorithms did was classifying the hospital where the data was collected instead of actually classifying the patient's disease.

Having a thorough understanding of your dataset is imperative for any data science task and while it should always come first in the workflow, it often doesn't, because it is just more work to analyze and plot the data than to hack together a small classifier and immediately start working on the actual task.
With Codaco, I want to change this so that the workflow *with* data exploration becomes the path of least resistance.

## State of the project: Barely started

Currently, I am just playing around with Python, getting up to date on all the neat data science libraries out there.

As a first working example, you can download the "abalone" database from the UCI machine learning database and view a histogram of the attributes with the following code:

```python
import codaco.data as cd
import codaco.stat as cs

data = cd.load_dataset("abalone", source="ucimlr")
cs.inspect_attributes(data)
```

## Why not Julia?

I debated whether I should use Julia or Python for this project, but decided on Python just because there are more ML-libraries for Python and because I am more familiar with the quirks of the language.
I might have had more fun if I used Julia, but I think I will get done more with Python.