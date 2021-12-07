([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L103))

The `Model` class is a subclass of `Layer`. For more details of how to
subclassing it to implement your own model, you may check out [this
tutorial](https://keras.io/guides/making_new_layers_and_models_via_subclassing/).

In the following workflow, the `Model` class is not so different from the
`Layer` class if you see it as a way to group the layers to build a
computational graph.

```py
class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)
  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)
```

However, it adds a set of functions and attributes that related to training, for
example. `compile()`, `fit()`, `evaluate()`, `predict()`, `optimizer`, `loss`,
`metrics`, which we would go into more details when we introduce the training
APIs. In summary, a `Model` can be trained by itself, but a `Layer` cannot.
