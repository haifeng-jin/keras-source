([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/optimizer_v2/optimizer_v2.py#L96))

In `Model.train_step()`, which we introduced in previous chapters,
`optimizer.minimize()` is called directly to update the trainable variables to
reduce the loss function value, while the gradients are recorded in the `tape`.
The pseudo-code is shown below.

```py
class Model(Layer):
    def train_step(self, data):
        x, y = data_adapter.unpack(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred)
        return {metric.name: metric.result() for metric in self.metrics}

```

To understand how this optimizer works, let's see what happens behind the
`.minimize()` function.

All optimizers in Keras extends the `OptimizerV2` class, which extends the
`Tractable` class. Remember that the `Layer` class also extends the
`Tractable` class. They all have variables to track. Any `tf.Variable` in its
attributes will be tracked automatically by TensorFlow.

In `OptimizerV2.minimize()`, which calls another method,
`OptimizerV2.apply_gradients()` to update the gradients. In `.minimize()` the
gradients are either obtained from the gradient tape, which is passed to it as
an argument or computed in the function using `loss` and `var_list`, which is a
list of trainable variables passed to it. When calling `.apply_gradients()`,
we zip the gradients and their corresponding variables into pairs and pass them
to it.

In `.apply_gradients()`, it updates the variables distributedly. The internal
function of `update_var()` is executed distributedly, which calls
`._resource_apply_dense()`, which is a function for the subclasses to override
to update the variable values with the gradients.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/optimizer_v2/optimizer_v2.py#L96))
```py
class OptimizerV2(Trackable):
    def minimize(self, loss, var_list, tape=None):
        tape = tape if tape is not None else tf.GradientTape()
        with tape:
            grads = tape.gradient(loss, var_list)
        self.apply_gradients(zip(grads, var_list))

    def apply_gradients(grads_and_vars):
        def update_var(var, grad):
            return self._resource_apply_dense(grad, var)
        strategy = tf.distribute.get_strategy()
        for grad, var in grads_and_vars:
            with strategy.extended.colocate_vars_with(var):
                distribution.extended.update(var, update_var, args=(grad, ))

    def _resource_apply_dense(self, grad, var):
        raise NotImplementedError
```

> **_TensorFlow API_** <br>
[`tf.GradientTape()`](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
Besides recording the tape using a `with` statement, it can also be used in a
stand-alone mode to return the gradient tape that automatically recorded the
gradients during the forward-pass by TensorFlow.

> **_TensorFlow API_** <br>
[`tf.distribute.StrategyExtended`](https://www.tensorflow.org/api_docs/python/tf/distribute/StrategyExtended)
The `strategy.extended` in the code example above is actually an instance of
`StrategyExtended`. All distribute strategies in TensorFlow have a `.extended`
attribute. It exposes some device and locality control of the variables and
tensors. For example, `.colocate_vars_with(var)` opens a scope where all the
newly created variables would be on the same device as `var`. `.update(var,
update_var, args=(grad, ))` run `update_var` to update `var` by mirroring the
args to the same device.

To make your own optimizer, you may need to override some of the functions, for
example, `._resource_apply_dense()`. Here is the pseudo-code for implementing
a stochastic gradient descent optimizer. We just override
`._resource_apply_dense()` and call the corresponding TensorFlow operation to
update the variables.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/optimizer_v2/gradient_descent.py#L23))
```py
class SGD(OptimizerV2):
    def _resource_apply_dense(self, grad, var):
        tf.raw_ops.ResourceApplyGradientDescent(var=var.handle, delta=grad)
```
Note that there are other functions to override to deal with other types of
tensors, for example, the sparse tensors.

> **_TensorFlow API_** <br>
[`tf.raw_ops`](https://www.tensorflow.org/api_docs/python/tf/raw_ops) This
`raw_ops` module in TensorFlow is a collection of raw C++ TensorFlow ops for
the user to directly use in Python. Each op is a series of tensor operations
that corresponds to a GPU kernel implemented in TensorFlow. Please refer to
[this guide](https://www.tensorflow.org/guide/create_op) for more details about
a TensorFlow op. It shows how to create a custom op.

