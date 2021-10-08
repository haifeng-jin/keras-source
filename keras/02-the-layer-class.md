## The `Layer` class

([Source](https://github.com/keras-team/keras/blob/r2.6/keras/engine/base_layer.py#L84))

The `Layer` class is easy to understand. It is the base class for the neural
network layers in Keras. It defines the forward pass of the computation in a
layer.

Here is a simple example of inheriting the `Layer` class to build a custom
layer.

```py
from tensorflow import keras

class SimpleDense(keras.layers.Layer):
  def __init__(self, units=32):
      super(SimpleDense, self).__init__()
      self.units = units
  def build(self, input_shape):
      self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
      self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)
  def call(self, inputs):
      return tf.matmul(inputs, self.w) + self.b
```

From this example, we can see that a `Layer` instance is a collection of
tensors and computations between the tensors in its attributes and the input
tensors. There are 4 methods to look into in the example. They are
`__init__()`, `build()`, `add_weight()`, and `call()`. The `__init__()`,
`build()`, and `call()` are expected to be overriden by the users, while
`add_weight()` is not. Let's see how they work one by one.

The `__init__()` function is easy to understand. It just records the arguments
from the caller with the attributes.

### The `Layer.build()` function

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/base_layer.py#L440))

The `build()` function is to create the `tf.Variable`s in the layer, which are
the weight and bias in the example above. Because the `tf.Variable`s are used
by the `call()` function, it would have to be created before the `call()`
function is called. Moreover, we don't want the variables to be created
multiple times. The question we want to answer here is how the build function
is called under the hood.

A lazy mechanism is implemented for `build()` with the `Layer._maybe_build()`
function, whose core logic is shown as follows. The `Layer` instance would use
the `self.built` attribute to record whether `build()` has been called. Any
code that would need the layer to be built would call this `_maybe_build()`
function to ensure the layer is built and built only once.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/base_layer.py#L2630))

```py
def _maybe_build(self, inputs):
  ...
  if not self.built:
    ...
    input_shapes = tf_utils.get_shapes(inputs)
    ...
    self.build(input_shapes)
    ...
  ...
```

The `tf_utils.get_shapes(inputs)` is a function in Keras to get the shapes of
the input tensors.

Here is an example of calling `_maybe_build()` secretly. We create a layer.
We call the layer with a tensor without explicitly calling `build()`.

```py
layer = SimpleDense(4)
layer(tf.ones((2, 2)))
```

Output:

```
<tf.Tensor: shape=(2, 4), dtype=float32, numpy=
array([[ 0.02684689, -0.07216483, -0.04574138,  0.03925534],
       [ 0.02684689, -0.07216483, -0.04574138,  0.03925534]],
      dtype=float32)>
```

The example runs successfully because the layer call would call the
`__call__()` function, which calls the `call()` function. Before calling the
`call()` function, `__call__()` would call `_maybe_build()` first to ensure the
`tf.Variable`s are created. The pseudo-code is shown as follows.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/base_layer.py#L1030))

```py
class Layer(module.Module, ...):
  def __call__(self, inputs, **kwargs):
    ...
    self._maybe_build(inputs)
    ...
    self.call(inputs)
    ...
```

This lazy pattern appears many times in Keras source code. When ensuring
something is called and don't want it to be called multiple times, you should
use this pattern. 


### The `Layer.add_weight()` function

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/base_layer.py#L528))

We would also like to see how these `tf.Variable`s are created in the
`add_weight()` function. Here is the pseudo-code for the core logic of the
`add_weight()` function.

It creates the variable and asks the backend to track the variable. The
variable will be appended to different lists depending on if it is trainable.

The [backend](https://github.com/keras-team/keras/blob/v2.6.0/keras/backend.py)
is a file containing some abstractions of the tensorflow functionalities. In
many cases, the Keras code would interact with the backend instead of directly
calling the TensorFlow APIs.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/base_layer.py#L528))

```py
class Layer(module.Module, ...):
  def add_weight(self, ...):
    ...
    variable = ...  # Create the variable.
    backend.track_variable(variable)  # Track the variable.
    if trainable:
      self._trainable_weights.append(variable)
    else:
      self._non_trainable_weights.append(variable)
```

The process for creating the variable is a [function
call](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/base_layer.py#L647),
which is not so different from using `tf.Variable(...)` to directly create the
variable.

We need the backend to track the variable for model saving and computation
optimization. Now, the question is how the backend is tracking the variable.
The code is shown as follows.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/backend.py#L1072))

```py
def track_variable(v):
  """Tracks the given variable for initialization."""
  if context.executing_eagerly():
    return
  graph = v.graph if hasattr(v, 'graph') else get_graph()
  _GRAPH_VARIABLES[graph].add(v)
```

We encountered two important concepts: [eager
mode](https://www.tensorflow.org/guide/eager) and [graph
mode](https://www.tensorflow.org/guide/intro_to_graphs). You can click the
link for detailed introductions.

Here is a short explanation. You can think of eager execution as plain Python
code execution. The tensors are all concrete values instead of placeholders.
The operations are read and executed only when we run that line of code in the
Python interpreter.

However, in graph mode, all the tensors and operations are collected in advance
to build the computational graph before any actual value is input for
computation. The graph is then optimized for execution speed. It is similar to
a compiled language, like the C programming language, which you can turn on
various optimization options to make the compiled executable file run faster.

> **_TensorFlow API_** <br>
`tf.executing_eagerly()` is to check whether TensorFlow is running in eager
mode or not.
([Link](https://www.tensorflow.org/api_docs/python/tf/executing_eagerly))

By default, everything runs in eager mode. As shown in the code above, in
eager mode, we don't need to track the variables because it would not compile
the computation graph.

In graph mode, the code above would record the `tf.Variable` to the
`_GRAPH_VARIABLES`, which is a dictionary mapping the TensorFlow computational
graphs to a list of `tf.Variable`s. With this dictionary, Keras can track all
the `tf.Variable`s for features like clearing the value of them.

### The `Layer.call()` function

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/base_layer.py#L926))

Let's see another use case of a layer. Instead of being part of a model, the
layer can directly be called to get the output. We can call the layer with a
Numpy array, it returns a `tf.Tensor` as the result.

```py
import tensorflow as tf
import numpy as np

layer = tf.keras.layers.Dense(input_shape=(10,), units=15)
x = np.random.rand(20, 10)
output = layer(x)
print(output.shape)  # (20, 15)
```

When we call `layer(x)`, it calls the `__call__()` function of the `Layer()`
class. The `__call__()` function would call the `call()` function, which
implements the forward pass of the layer.

Calling the layer with a Numpy array, the `__call__()` function will just
convert it to a `tf.Tensor` and call the `call()` function. If in eager mode,
the function can be directly called. If in graph mode, we will need to convert
the function to a computational graph before calling it.

> **_TensorFlow API_** <br>
`tf.function()` is the public API in TensorFlow for converting a normal
function into a computation graph.
([Link](https://www.tensorflow.org/api_docs/python/tf/function))

There is another use case that quite separated from the use case above, which
is calling the layer with a
[`KerasTensor`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/keras_tensor.py#L30).
It is a class used when creating models using the [functional
API](https://keras.io/guides/functional_api/). It is a symbolic tensor without
actual value, but only representing the shapes and types of intermediate output
tensors between the layers. We will introduce more about it when we introduce
the `Model` class.

The pseudo-code for the `__call__()` function is shown as follows.

```py
class Layer(module.Module, ...):

  def __call__(self, inputs, **kwargs):

    if isinstance(inputs, keras_tensor.KerasTensor):
      inputs = convert_to_tf_tensor(inputs)
      outputs = self.call(inputs)
      return convert_to_keras_tensor(outputs)

    if isinstance(inputs, np.ndarray):
      inputs = tf.Tensor(inputs)

    # Check the inputs are compatible with the layer.
    input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)

    if context.executing_eagerly():
      return self.call(inputs)
    else:
      call_fn = convert_to_tf_function(self.call)
      return call_fn(inputs)
```


### Input compatibility checking

Notably, in the code above, the inputs compatibility is being checked before
the actual call. This is to ensure the bug is being caught early and throw
out a meaningful error message. If you run the following code, you will get an
error. It calls the `Dense` layer with vectors of length 5 first, which causes
the layer to initialize the weights for these vectors. Then, it calls the same
layer again with vectors of length 4, which are not compatible with the weights
created just now.

```py
layer = layers.Dense(3)
layer(np.random.rand(10, 5))
layer(np.random.rand(10, 4))
```

Error message:

```
ValueError: Input 0 of layer dense is incompatible with the layer: expected axis -1 of input shape to have value 5 but received input with shape (10, 4)
```

The inputs compatibility information is recorded in `self.input_spec` of a
layer, which is a function with
[`@property`](https://docs.python.org/3/library/functions.html#property)
decorator. Now, we would like to see how are the `input_spec` being recorded
and used to check new inputs.

The base `Layer` class doesn't record this `input_spec`. Each layer should
record their own `self.input_spec` in `build()`. In `build()`, they usually
need to create the weights of the layer using the `input_shape`. Meanwhile,
they can define a `self.input_spec` for what inputs are compatible with these
weights. For example, in `Dense.build()`, they set the `input_spec` for a
fixed last dimension by creating an `InputSpec` instance
([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/layers/core.py#L1177)).

The signature of the `InputSpec` is as follows. There are many specifications
to set, like the shape, data type, number of dimensions, upper and lower bound
of dimensions.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/input_spec.py#L29))

```py
class InputSpec(object):
  def __init__(self,
               dtype=None,
               shape=None,
               ndim=None,
               max_ndim=None,
               min_ndim=None,
               axes=None,
               allow_last_axis_squeeze=False,
               name=None):
```

In `__call__()`, `assert_input_compatibility()` would check all the
specifications of the recorded `self.input_spec` for the given inputs for
compatibility.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/input_spec.py#L154))
