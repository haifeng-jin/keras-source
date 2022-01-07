([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/metrics.py#L81))

All metrics in Keras are derived from the
[`Metric`](https://github.com/keras-team/keras/blob/v2.6.0/keras/metrics.py#L81)
class.

A metric can be used directly instead of passing it to a model as shown in the
following code example. We call the `.update_state()` function multiple times
to pass the `y_true` and `y_pred` in different batches to it. Then, use
`.result()` to get the metric value. We can also use `.reset_state()` to clear
all previous computed values.

```py
mse = keras.metrics.MeanSquaredError()

mse.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
mse.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
print(mse.result().numpy())  # 0.25
mse.reset_state()
mse.update_state([[0, 1]], [[1, 1]])
print(mse.result().numpy())  # 0.5
```

When subclassing the `Metric` class, one needs to override `.update_state()`,
`.result()`, and `.reset_state()`. Refer to
[this tutorial](https://keras.io/getting_started/intro_to_keras_for_researchers/#keeping-track-of-training-metrics)
for more details on how to implement a custom metric.

To keep track of all the statistics in the metric, which are all
`tf.Variable`s, `Metric` extends the `Layer` class. The `.update_state()` is
compiled into a `tf.function` for faster computation. The pseudo code of the
`Metric` class is shown as follows.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/metrics.py#L163))

```py
class Metric(Layer):
    def __new__(self, cls, *args, **kwargs):
        obj = super(Metric, cls).__new__(cls)
        obj.update_state = tf.function(obj.update_state)
        return obj

    def update_state(self):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def reset_state(self):
        raise NotImplementedError
```

Take notes of the following subclasses of `Metric`.
[`Reduce`](https://github.com/keras-team/keras/blob/v2.6.0/keras/metrics.py#L361)
is a metric that computes a single value out of a tensor. The computation is
defined by an argument passed to the initializer. For example, it can be
computing the sum or mean of all the values in the tensor.

[`Mean`](https://github.com/keras-team/keras/blob/v2.6.0/keras/metrics.py#L497)
is a subclass of `Recude`, which just compute the mean of the values in the
tensor.

[`MeanMetricWrapper`](https://github.com/keras-team/keras/blob/v2.6.0/keras/metrics.py#L619)
is a subclass of `Mean`. It is similar to `LossFunctionWrapper` introduced in
the previous section. It converts a metric function into a `Metric` subclass.
It extends `Mean` because all metric functions needs to be averaged across
batches.

In `Model.compile()`, all the metrics are wrapped up into a single
[`MetricsContainer`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/compile_utils.py#L289)
instance. Similar to the `LossesContainer`, it encapsulates all the metrics to
be easily used by the `Model` class. It implements `.update_state()` and
`.reset_state()` just like a `Metric` subclass so that the `Model` class will
use this `MetricsContainer` just like a single metric. It doesn't need to
implement `.result()` because the result of each metric is displayed
separately. During initialization, it converts the metric strings or functions
to `Metric` subclass instances.

Notably, the metrics `MetricsContainer` receives is a list of lists of metrics
because each head of the neural network model has a list of metrics.

The pseudo code of `MetricsContainer` is shown as follows.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/compile_utils.py#L289))
```py
class MetricsContainer(Container):

    def __init__(self, metrics):
        self._metrics = [self._get_metric_object(metric) for metric in metrics]

    def update_state(self, y_true, y_pred):
        # y_pred is a list of outputs.
        # Each element in the list is the output of one of the heads.
        # y_true is the ground truth for the heads in a similar format.
        for single_y_true, single_y_pred, single_metrics in zip(
            y_true, y_pred, self._metrics):
            # Iterate the metrics for the current head.
            for metric_obj in single_metrics:
                metric_obj.update_state(single_y_true, single_y_pred)

    def reset_state(self):
        for metric_obj in tf.nest.flatten(self._metrics):
            metric_obj.reset_state()

    def _get_metric_objects(self, metric):
        # get() may return a function instead of a Metric instance.
        metric_obj = keras.metrics.get(metric)
        if not isinstance(metric, keras.metrics.Metric):
            metric_obj = keras.metrics.MeanMetricWrapper(metric_obj)
        return metric_obj
```
