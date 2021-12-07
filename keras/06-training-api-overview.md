From here on, we will start to introduce how the training process works.
We will try to understand what happends behind the scene of the following code.

```py
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
model.fit(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    epochs=2)
```

We will introduce the following items:
* How the `Model.compile()` function works.
* How the `Model.fit()`, `Model.predict()`, and `Model.evaluate()` function works.
* How the TensorFlow distributed training API is used in training.
* How are the optimizer, loss, and metrics implemented and used.
