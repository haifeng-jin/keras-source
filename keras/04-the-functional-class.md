### The `Functional` class

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/functional.py#L41))

There is another way of using the `Model` class besides subclassing it, which
is the functional API. It connects the layers to each other to form a directed
acyclic graph (DAG), where the nodes are layer call events, and the edges are
KerasTensors. Please refer to [this
tutorial](https://keras.io/guides/functional_api/) for more details of how to
use it. Following is a code example of using the functional API.

```py
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

Although it looks like it is still using the `Model` class, it is 
using the `Functional` class, which is an internal class not exposed to the
public API. In `Model.__new__()`, it creates a `Functional` instance if using
the functional API. The source code looks like this.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/training.py#L189))

```py
class Model(Layer):
  def __new__(cls, *args, **kwargs):
    if is_functional_model_init_params(args, kwargs) and cls == Model:
      return functional.Functional(skip_init=True, *args, **kwargs)
    ...
```

Now, let's see how `Functional` is tracking these layers and intermediate
outputs in the computational graph.

#### The `KerasTensor` class

`keras.Input()`, which looks like a class, but it is a function, which
returns a `KerasTensor` object.

```py
print(type(keras.Input(shape=(28, 28, 1))))
```

Outputs:

```
<class 'keras.engine.keras_tensor.KerasTensor'>
```

`KerasTensor` is a class just to represent the intermediate output tensors of
the layers in a Keras model, which has some useful properties like `shape` and
`dtype`.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/keras_tensor.py#L30))

```py
class KerasTensor(object):
  @property
  def shape(self):
    ...
  @property
  def dtype(self):
    ...
```

It is passed to each of the layers by calling them as shown in the functional
API example. The purpose is for the layers to create the weights using the
shape and type information of the input tensor. That is also why we have a
special judge to see if the input tensor is a `KerasTensor` in
`Layer.__call__()` as we introduced before.

```py
class Layer(module.Module, ...):
  def __call__(self, inputs, **kwargs):
    if isinstance(inputs, keras_tensor.KerasTensor):
      inputs = convert_to_tf_tensor(inputs)
      outputs = self.call(inputs)
      return convert_to_keras_tensor(outputs)
    ...
```

From the source code above, we can see if we call a layer with a `KerasTensor`,
the return value is also a `KerasTensor`, which will be used to call the next
layer.

#### Connecting the layers

The question we try to answer here is: How did the computational graph being
recorded and fetched only given the `inputs` and `outputs`? This functionality
is implemented in
[`Functional._init_graph_network()`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/functional.py#L112).

The graph is being fetched starting from the `outputs`, which is a list of
`KerasTensor`s. Each `KerasTensor` records the `Layer` instance that produces
it during the call of the `Layer`. The algorithm is like this. First, from
the `outputs`, we got the `Layer` producing these `outputs`. Second, use 
the `Layer` to get the input `KerasTensors`. Third, use these `KerasTensor`s to
get the previous layers. Keep doing this until the inputs to the model are
reached.

Here are another two questions to answer:

1. How does a `KerasTensor` get the `Layer` producing it?

2. How does the `Layer` get the input `KerasTensor`s?

To answer these two questions, there are two important classes or concepts to
make clear first:
[`Node`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/node.py#L33)
and
[`KerasHistory`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/node.py#L261).

A `Node` is created at each call of a layer, to represent the connectivity
between the two `Layer`s. In other words, a `Node` corresponds to a call of a
`Layer`. Each `Layer` has an attribute of
[`_inbound_nodes`](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/base_layer.py#L2258)
to track the input `Node`s. The reason why there can be multiple inbound
`Node`s is that a `Layer` may be used in multiple places in a model for weight
sharing.

In the following example, `layer_a` and `layer_b` are all called multiple
times. Therefore, `node1` and `node4` are in `layer_a._inbound_nodes`.
`node2` and `node5` are in `layer_b._inbound_nodes`.

`node1 -> layer_a -> node2 -> layer_b -> node3`

`node4 -> layer_a -> node5 -> layer_b -> node6`

When building a functional model, we can call a `Layer` with multiple
`KerasTensor`s. For example, the
[`Add`](https://keras.io/api/layers/merging_layers/add/) layer add multiple
tensors together. Therefore, a call of a layer corresponds to multiple
`KerasTensor`s. A `Node` also corresponds to a call of a `Layer`. Therefore,
a `Node` may correspond to multiple `KerasTensor`s. A `Node` record these
`KerasTensor`s in `Node._keras_inputs`.

`KerasHistory` is for the `KerasTensor` to find the input `Layer` and the
corresponding `Node`. It is stored in the attribute of
`KerasTensor._keras_history`. `KerasHistory.layer` records the `Layer`
producing it. `KerasHistory.node_index` records the index of the corresponding
`Node` in the `KerasHistory.layer._inbound_nodes`.

For example, if `node2` has a corresponding `KerasTensor`, named
`keras_tensor_2`, `keras_tensor_2._keras_history.node_index` records the index
of `node1` in `layer_a._inbound_nodes`.

With all these recording mechanisms, we can have the following pseudo-code.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/base_layer.py#L2591))

```py
class Layer(tf.Module):
  def __call__(inputs):
    ...
    outputs = self.call(inputs)
    node = Node(self, inputs, outputs)
    ...
```

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/node.py#L96-L100))

```py
class Node:
  def __init__(self, layer, inputs, outputs):
    ...
    self._keras_inputs = inputs
    layer._inbound_nodes.append(self)
    node_index = len(self.layer._inbound_nodes) - 1
    for keras_tensor in outputs:
        keras_tensor._keras_history = KerasHistory(layer, node_index)
    ...
```

Now, we have the answers to the two questions above. A `KerasTensor` find the
layer producing it with `KerasTensor._keras_history.layer`. A `Layer` find the
input `KerasTensor` with `Layer._inbound_nodes[0]._keras_inputs`.

Finally, we can fetch the entire computational graph with the algorithm
described above. The pseudo-code is shown as follows.

([Source](https://github.com/keras-team/keras/blob/v2.6.0/keras/engine/functional.py#L1031))

```py
def fetch_nodes(output_keras_tensor):
  if output_keras_tensor in model.inputs:
    return []
  layer = output_keras_tensor._keras_history.layer
  node_index = output_keras_tensor._keras_history.node_index
  node = layer._inbound_nodes[node_index]
  node_list = [node]
  for input_keras_tensor in node._keras_inputs:
    node_list += fetch_nodes(input_keras_tensor)
  return node_list
```

The actual code would not only fetch the nodes, but also the layers, and sort
them in topological order. In `Functional.call`, it calls the layer in
topological order to produce the outputs.
