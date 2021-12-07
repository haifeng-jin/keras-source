([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L876))

The first thing the `Model.fit()` function does is to convert the user passed
dataset (`x` and `y`) into a compatible format that ready to be used for the
training. They are wrapped into a
[`DataHandler`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/data_adapter.py#L1080),
which can be used conveniently during the training. It has a method, named
[`enumerate_epochs()`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/data_adapter.py#L1177),
which returns the current epoch number and the iterator of the dataset. It
also has a method, named
[`steps()`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/data_adapter.py#L1229),
which returns the current step number. These two functions are mainly used by
the `Model.fit()` function to iterate over the dataset for each epoch.

In side `DataHandler`, it would convert different types of data into a
`tf.data.Dataset` object with different
[`DataAdapter`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/data_adapter.py#L40)s.
That is how Keras supports so many different types of data inputs.

With `DataHandler`, we prepared the dataset to be iterated batch-by-batch. For
each batch of data, we would do a forward pass and updates of the weights,
which is called a step in an epoch. We want to build a function to execute a
single step, and compile it into a `tf.function` to accelerate the process.
Here we use
[`Model.make_train_function()`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L803)
to get the function.

Now we can use the following pseudo code to summarize what happend in
`Model.fit()`. We first wrap the data into a `DataHandler`. Get the
`tf.function` for running a step. Use a for loop to iterate through the
epochs, and use an inner for loop to iterate the steps. In each step, we just
call the function to execute the step.

```py
class Model(Layer):
    def fit(self, x, y, ...):
        data_handler = DataHandler(x, y)
        self.train_function = self.make_train_function()
        for epoch, iterator in data_handler.enumerate_epochs():
            for step in data_handler.steps():
                self.train_function(iterator)
```

## Train step

Now, it comes to the question of what happens in `Model.make_train_function()`.
You can think it just returns the
[`Model.train_step()`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L755)).
Notebly, the user can also override this `Model.train_step()` function to
[customize their training
step](https://keras.io/guides/customizing_what_happens_in_fit/).

Following is the pseudo code for `Model.train_step()`. It runs the forward
pass using `self(x)` and compute the loss value, while recording all the
gradients with `tf.GradientTape()`. Then, it use the `optimizer` to minimize
the loss function to update the trainable variables using the gradients.
Finally, it returns the metrics.

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

## Distributed training

Actually, `Model.make_train_function()` adds one more functionality to
`Model.train_step()`, which is to support the distributed training. Let's see
how distributed training is supported in Keras with the TensorFlow APIs.

First, we need to use `tf.distribute.Strategy.scope()`, which opens up a scope
to track all the TensorFlow variables created in this scope, for example the
weights of the neural network.

> **_TensorFlow API_** <br>
[`tf.distribute.Strategy.scope()`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#scope).
`scope()` opens up a scope that any `tf.Variable()` created inside the scope is
caught by TensorFlow to run distributedly.

To ensure everything is caught by the distributed strategy, we need to put
almost the entire `Model.fit()` function in the scope as shown in the following
pseudo code.

```py
class Model(Layer):
    def fit(self, x, y, ...):
        with self.distribute_strategy.scope():
            data_handler = data_adapter.get_data_handler(x, y)
            self.train_function = self.make_train_function()
            for epoch, iterator in data_handler.enumerate_epochs():
                for step in data_handler.steps():
                    self.train_function(iterator)
```

Another TensorFlow API we need to use here is `tf.distribute.Strategy.run()`.
When run distributedly, the `Model.train_step()` function needs to run on each
replica in parallel. The `Model.make_train_function()` function wraps
`Model.train_step()` into another function that uses
`tf.distribute.Strategy.run()` to run `train_step()` distributedly. It also
convert the function into a `tf.function`.

> **_TensorFlow API_** <br>
[`tf.distribute.Strategy.run()`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#run)
runs a function on each replica with the given arguments. For example,
`strategy.run(fn, args=(arg1, arg2))`.

The pseudo code of wrapping `Model.train_step()` is as follows. We wrap
`.train_step()` into a new function, `train_function()`, convert it to a
`tf.function`, and return it to `.fit()`. In `train_function()`, we just call
`.train_step()` using `distribute.Strategy.run()` and aggregate the outputs and
return it.

```py
class Model(Layer):
    def make_train_function(self, ...):
        def train_function(iterator):
            data = next(iterator)
            outputs = model.distribute_strategy.run(self.train_step, args=(data,))
            outputs = reduce_per_replica(outputs)
            return outputs
        train_function = tf.function(train_function)
        return train_function
```

There is another TensorFlow distribute strategy API that is used by Keras is
`tf.distribute.get_strategy()`. That is how Keras get the distribute strategy
defined by the user.

> **_TensorFlow API_** <br>
[`tf.distribute.get_strategy()`](https://www.tensorflow.org/api_docs/python/tf/distribute/get_strategy)
Returns the current tf.distribute.Strategy object.
