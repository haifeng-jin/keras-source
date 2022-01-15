([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L35))

All loss functions implemented in Keras are subclasses of the
[`Loss`](https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L35)
class.

For example, you can implement a `MeanSquaredError` loss as follows.

```python
class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)
```

You can use it as a standalone function as follows. `Loss.call()` is called by
`Loss.__call__()` under the hood.

```python
loss = MeanSquaredError()
print(loss(np.array([0.0, 1.0]), np.array([1.0, 1.0])))  # tf.Tensor(0.5, shape=(), dtype=float64)
```

All the built-in losses are implemented in a similar way, which is to override
the `call()` function. It computes the loss for the given ground truth and
predictions.

As you may know, the loss passed to `.compile()` can be a string, a function, or
a `Loss` subclass instance. However, they are all converted to a `Loss`
subclass in the end. In `.comiple()`, it uses
[`keras.losses.get()`](https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L2099)
to convert the losses. The `get()` function can accept a string, which is the
name of the loss, and returns either a loss function or a `Loss` subclass
instance. If it is a function, Keras would further convert it to a `Loss`
subclass instance using the `LossFunctionWrapper`, which wraps the function
into a `Loss` subclass instance. The overall converting process is shown in
the following pseudo-code.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/compile_utils.py#L273-L278))
```py
def get_loss(loss):
    loss = keras.losses.get(loss)
    # If it is an function
    if not isinstance(loss, keras.losses.Loss):
        loss = losses_mod.LossFunctionWrapper(loss, name=loss_name)
    return loss
```

The `LossFunctionWrapper` class is just a subclass of `Loss` and calls the
provided loss function in its `.call()` function as shown in the following
pseudo-code. We also show an example of wrapping up a `mean_squared_error()`
loss function into a `Loss` subclass instance.

```py
class LossFunctionWrapper(Loss):
    def __init__(self, fn):
        self.fn = fn

    def call(self, y_true, y_pred):
        return self.fn(y_true, y_pred)

def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)

loss = LossFunctionWrapper(mean_squared_error)
```

It is a common pattern in Keras, which wraps a function into a class instance.
We will see this pattern again in metrics as well, which will be introduced
later.

As you can see from above, the `Loss.call()` function is only dealing with a
single batch of data. During training, computing the loss of a single batch of
data is enough for backpropagation. However, we also need to compute the
average loss value of all the trained batches to print to screen during
training. It is done by the
[`LossesContainer`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/compile_utils.py#L100)
class, which we mentioned when introducing `.compile()`. It manages a metric
to track the historical loss values for the batches. Unlike a loss function,
the metric can not only compute the metric for one batch of data but also
record some statistics across the historical batches. We will introduce more
details about how metrics work in Keras later.


The `LossesContainer` class also manages multiple losses, which is supported in
Keras. The user can pass multiple losses to a model with multiple heads, where
each loss corresponds to one head. The `LossesContainer` class contains all
these losses and exposes methods that can manage them as if they are a single
object.

This is also a pattern. Using containers to manage a collection of objects and
behaves similarly to a single object. We will see it again when we introduce
the metrics.

The only method of `LossesContainer` that directly called by the `Model` class
is `.__call__()`. As we introduced in `.fit()` part, `.train_step()` use
`self.compiled_loss(y, y_pred)` to compute the loss value.

In `LossesContainer.__call__()`, it iterates through the different heads of the
model and computes the losses, and sums them up. The pseudo-code is as follows.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/compile_utils.py#L161))
```py
class LossesContainer(Container):
    def __call__(self, y_true, y_pred):
        # y_pred is a list of outputs.
        # Each element in the list is the output of one of the heads.
        # y_true is the ground truth for the heads in a similar format.
        for single_y_true, single_y_pred, single_loss in zip(
            y_true, y_pred, self._losses):
            loss_values.append(
                single_loss(single_y_true, single_y_pred))
        total_loss = sum(loss_values)
        # Updating the metric tracking the average loss value
        # across the historical batches.
        self._losses_metric.update_state(total_loss)
        return total_loss
```
