([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L1610))

The logic of the `Model.predict()` function is very similar to `Model.fit()` as
shown in the following pseudo code. It first wraps the input `x` into a
`DataHandler`. Then, build the
[`Model.predict_step()`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L1515)
into a `tf.function` with
[`Model.make_predict_function()`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L1539).
`.predict_step()` make predicitons for a single batch of data, which can also
be overridden to customize the predict behavior. Similar to
`.make_train_function()`, `.make_predict_function()` would handle the
distribute strategy while building the `tf.function`.

```py
class Model(Layer):
    def predict(self, x, ...):
        data_handler = DataHandler(x)
        self.predict_function = self.make_predict_function()
        outputs = []
        for epoch, iterator in data_handler.enumerate_epochs():
            for step in data_handler.steps():
                outputs.append(self.predict_function(iterator))
        return outputs
```

By default, the `Model.predict_step()` function would just unpack `x` from the
provided `data` (because `data` may contain `y`) and call the model to do a
forward pass with the data as shown in the following pseudo code.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L1515))

```py
class Model(Layer):
    def predict_step(self, data):
        x = data_adapter.unpack(data)
        return self(x, training=False)
```
