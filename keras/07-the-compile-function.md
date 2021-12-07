([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L449))

We usually need to call `Model.compile()` before we train the model, as shown
in the following code example.

```py
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

As you expected, the `Model.compile()` function is just recording these
configurations. Since the user may provide the `optimizer` as a string, we use
`get_optimizer()` function to get the corresponding
[`keras.optimizers.Optimizer`](https://github.com/keras-team/keras/blob/v2.6.0/keras/optimizer_v2/optimizer_v2.py#L96)
instance.

The `loss` and `metrics` can be lists or dictionaries of loss functions and
metrics. Threrefore, we need to encapsulate them into data structures, which
are easier to use, which can be treated as single objects instead of using a
for loops to deal with each of the losses or metrics.

The core functionality of `Model.compile()` is shown as in the following pseudo
code.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L558-L563))

```py
class Model(Layer):
    def compile(self, loss, optimizer, metrics, ...):
        self.optimizer = get_optimizer(optimizer)
        self.compiled_loss = LossesContainer(loss)
        self.compiled_metrics = MetricsContainer(metrics)
```

Besides the loss, optimizer, and metrics, which are the most important
configurations for the training, there are other interesting configurations in
teh compile function as well, you may try to explore them in the source code by
yourself.
