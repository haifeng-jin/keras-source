([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/sequential.py#L41))

The `Sequential` class extends the `Functional` class. It mainly supports a
special case of the `Functional` model, where only a single chain of layers in
the model without any branches. For more details of how to use it, you can
check out [this tutorial](https://keras.io/guides/sequential_model/).

It implements the `add()` method and the `pop()` method to easily handle adding
an removing the layers.

`Sequential` has two ways to build the model depending whether the
`input_shape` of the model is know from the beginning.

In the following example, the model knows the `input_shape` from the beginning.
It just treats the model as a `Functional` model.

```py
model = keras.Sequential() model.add(keras.Input(shape=(10,)))
model.add(keras.layers.Dense(units=10, activation='relu'))
model.add(keras.layers.Dense(units=1))
```

However, in the following example, the model would not know the `input_shape`
until it sees the first batch of training data. Therefore, the initialization
of the computational graph is deferred.

```py
model = keras.Sequential()
model.add(keras.layers.Dense(units=10, activation='relu'))
model.add(keras.layers.Dense(units=1))
```

The pseudo-code for checking the two cases is as follows.

```py
class Sequential(Functional):
  def add(self, layer):
    ...
    if self._has_input_shape:
      # This is the funciton used by `Functional`
      # to build the computational graph.
      self._init_graph_network(self.inputs, self.outputs)
    else:
      self.layers.append(layer)
    ...

  def call(self, inputs, ...):
    if not self._has_input_shape:
      self._build_graph_network(inputs.shape)
    ...
```

### Summary

So far, we have gone through the framework of all the code for model building.
We have introduced the chain of extension, from `tf.Module` to `Sequential`,
what functionalities are added in each subclass along the way. We also
introduced some important concepts, like eager mode, graph mode, `Tensor`,
`Variable`, `KerasTensor`, and `Node`. We also introduced some important
mechanisms, like the `@keras_export`, `_maybe_build()` to ensure the model is only
being built for once, creating and tracking the weights, `InputSpec` checking,
computational graph fetching in `Functional`.

Next, we will introduce the source code of the training APIs of Keras. We will
see how does `Model.compile()` and `Model.fit()` works, how the loss are being
tracked, how the optimizer updates the weights, and so on.
