([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L1352))

The logic of the `Model.evaluate()` function is very similar to `Model.fit()`
and `Model.predict()` as shown in the following pseudo code. It first wraps the
input `x` and `y` into a `DataHandler`. Then, build the
[`Model.test_step()`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L1241)
into a `tf.function` with
[`Model.make_test_function()`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L1282).
`.test_step()` do evaluations for a single batch of data, which can also be
overridden to customize its behavior. Similar to `.make_train_function()` and
`.make_predict_function()`, `.make_test_function()` would handle the distribute
strategy while building the `tf.function`.

```py
class Model(Layer):
    def evaluate(self, x, y, ...):
        data_handler = DataHandler(x, y)
        self.test_function = self.make_test_function()
        logs = {}
        for epoch, iterator in data_handler.enumerate_epochs():
            for step in data_handler.steps():
                # Always record the last epoch's metric.
                logs = self.test_function(iterator)
        return logs
```

By default, the `Model.test_step()` function would just unpack `x` and `y` from
the provided `data` and call the model to do a forward pass with the data to
get the predictions. Then, it uses the prediction and the ground truth `y` to
compute the metric values to return. The pseudo code is shown as follows.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L1515))

```py
class Model(Layer):
    def predict_step(self, data):
        x = data_adapter.unpack(data)
        y_pred = self(x, training=False)
        self.compiled_metrics.update_state(y, y_pred)
        return_metrics = {}
        for metric in self.metrics:
            return_metrics[metric.name] = metric.result()
        return return_metrics
```
