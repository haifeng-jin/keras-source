The modeling API of Keras is responsible for putting the layers or TensorFlow
operations together to create a model before training.

### A chain of class inheritance

Here is a simple [Sequential model](https://keras.io/guides/sequential_model/)
example, which creates a model with minimal code.

```py
import keras

model = keras.Sequential()
model.add(keras.layers.Dense(input_shape=(10,), units=10, activation='relu'))
model.add(keras.layers.Dense(units=1))
```

The `Sequential` class is a high-level class, which has a chain of base
classes. The inheritance chain looks like this:

`tf.Module -> Layer -> Model -> Functional -> Sequential`

The class on the left of an arrow is the base class of the one on the right.
To understand this chain of classes, we start from the base classes to see what
functionality has been added step-by-step by the subclass down the chain.

### The `tf.Module` class 

([Source](https://github.com/tensorflow/tensorflow/blob/v2.6.0/tensorflow/python/module/module.py#L35))

The first base class to dive into is the `tf.Module` class, which is a core
class in TensorFlow. It is used by Keras. You can think of it as a container for
`tf.Variable` instances.

A [`tf.Variable`](https://www.tensorflow.org/guide/variable) instance is a data
structure for storing a mutable tensor in TensorFlow. The difference between a
`tf.Variable` and a [`tf.Tensor`](https://www.tensorflow.org/guide/tensor) is
that `tf.Variable` is mutable but `tf.Tensor` is not. The weight of a layer is
a `tf.Variable` instance.

A typical usage of the `tf.Module` class is to group a series of operations on
the tensors, for example, a neural network layer. It has an
attribute called `name_scope`, which is used as the prefix for the names of
its `tf.Variable` instances.

```py
import tensorflow as tf

constant_tensor = tf.constant([10, 20, 30])
class MyModule(tf.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    with self.name_scope:  # Open the name space.
      self.variable = tf.Variable(
          constant_tensor, name="my_variable")  # Create in the name space.

# The format of the name is "`module_name/variable_name:counter`"
print(MyModule(name="my_module").variable)
```

Output:

```
<tf.Variable 'my_module/my_variable:0' shape=(3,) dtype=int32, numpy=array([10, 20, 30], dtype=int32)>
```

However, the name of a `tf.Variables` is only an internal identifier of the
object, which should not be used directly by the users.

Because it is not part of the Keras codebase, we will not dive into the
implementation of it. Another feature of the class is that it inherits the
`Trackable` class, which tracks all the `tf.Variable` instances in the
attributes of the subclasses of `tf.Module`. When saving the models, all the
`tf.Variable` instances inside this container can be found and saved. The
variables are also tracked for optimizing the computational graph.

### File locations

Before we show how the `Layer` class works, let's first see where the code
of the base `Layer` class and the subclasses is, like `Conv2D`. This
is a typical case for the organizing code in Keras. The base `Layer` class
is in `/keras/engine/base_layer.py`, while the subclasses are in the
`/keras/layers` directory.

The base classes, which builds the Keras overall framework, are in the
`/keras/engine` directory. The implementation of each of the subclasses is in
its corresponding directory. With this file location logic, you can
navigate through the codebase.

### The import mechanism

The importing path of the `Layer` class is `tf.keras.layers.Layer`. However,
the code of the class is in `/keras/engine/base_layer.py`. It is designed to
decouple the importing path and the actual path to give better flexibility for
the implementation. This import mechanism is implemented with the
[`@keras_export()`](https://github.com/tensorflow/tensorflow/blob/v2.6.0/tensorflow/python/util/tf_export.py#L411)
decorator, which is implemented using the
[`@tf_export()`](https://github.com/tensorflow/tensorflow/blob/v2.6.0/tensorflow/python/util/tf_export.py#L409)
decorator. With `@keras_export('keras.layers.Layer')`, the 'Layer' class can
be imported from `tf.keras.layers.Layer`.
