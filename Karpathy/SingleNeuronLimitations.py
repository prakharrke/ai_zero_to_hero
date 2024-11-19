import marimo

__generated_with = "0.9.16"
app = marimo.App()


app._unparsable_cell(
    r"""
    !pip install numpy micrograd matplotlib
    """,
    name="__"
)


@app.cell
def __():
    import math 
    import numpy as np
    import matplotlib.pyplot as plt
    # %matplotlib inline
    return math, np, plt


@app.cell
def __(math):
    class Value:

        def __init__(self, x, _children=set(), _op='', label=''):
            self.data = x
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op
            self.label = label

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), '+')

            def _backward():
                self.grad = self.grad + 1.0 * out.grad
                other.grad = other.grad + 1.0 * out.grad
            out._backward = _backward
            return out

        def __radd__(self, other):
            return self + other

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), '*')

            def _backward():
                self.grad = +other.data * out.grad
                other.grad = +self.data * out.grad
            out._backward = _backward
            return out

        def __rmul__(self, other):
            return self * other

        def __neg__(self):
            return self * -1

        def __sub__(self, other):
            return self + -other

        def __pow__(self, other):
            out = Value(self.data ** other, (self,), label=f'**{other}')

            def _backward():
                self.grad = self.grad + other * self.data ** (other - 1) * out.grad
            out._backward = _backward
            return out

        def __repr__(self):
            return f'Value=(data={self.data})'

        def exp(self):
            x = self.data
            out = Value(math.exp(x), (self,), label='exp')

            def _backward():
                self.grad = out.data * out.grad
            self._backward = _backward
            return out

        def __truediv__(self, other):
            return self * other ** (-1)

        def tanh(self):
            x = self.data
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
            out = Value(t, (self,), 'tanh')

            def _backward():
                self.grad = self.grad + (1 - t ** 2) * out.grad
            out._backward = _backward
            return out

        def backward(self):
            topo = []
            visited = set()

            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)
            self.grad = 1
            for v in reversed(topo):
                v._backward()
    return (Value,)


@app.cell
def __():
    from graphviz import Digraph

    def trace(root):
      # builds a set of all nodes and edges in a graph
      nodes, edges = set(), set()
      def build(v):
        if v not in nodes:
          nodes.add(v)
          for child in v._prev:
            edges.add((child, v))
            build(child)
      build(root)
      return nodes, edges

    def draw_dot(root):
      dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

      nodes, edges = trace(root)
      for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
          # if this value is a result of some operation, create an op node for it
          dot.node(name = uid + n._op, label = n._op)
          # and connect this node to it
          dot.edge(uid + n._op, uid)

      for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

      return dot
    return Digraph, draw_dot, trace


@app.cell
def __(plt):
    import matplotlib.image as mpimg
    img=mpimg.imread('neuronMathModel.png')
    plt.imshow(img)
    return img, mpimg


@app.cell
def __():
    # Inputs to a neuron are called axons and mathematically they can be x0, x1... xn. Inputs come to a neuron through synapses. 
    # Each synapse has a weight associated with it. When an input x0 comes to neuron through a synapse s with weight w0, the weight of synapse w0 gets multiplied to x0. 

    # If a neuron has n synapses and n inputes are sent through those n synapses, overall input to a neuron becomes:
    # x0w0 + x1w1 + .... + xnwn
    # Bias
    # Bias of a neuron is the 'happy trigger' that is applies to the output
    # So the neuron will be applying bias be to the afformentioned input. (x0w0 + x1w1 + ... + nxwn) + b
    # Apply the activation function to the output of n to get the final output of the neuron
    return


@app.cell
def __(Value):
    # inputs x1,x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights w1,w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bias of the neuron
    b = Value(6.8813735870195432, label='b')
    # x1*w1 + x2*w2 + b
    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    return b, n, o, w1, w2, x1, x1w1, x1w1x2w2, x2, x2w2


@app.cell
def __(o):
    o.backward()
    return


@app.cell
def __(draw_dot, o):
    draw_dot(o)
    return


@app.cell
def __(Value):
    import random
    class Neuron: 
        def __init__(self, n):
            self.w = [Value(random.uniform(-1, 1)) for _ in range(n)]
            self.b = Value(random.uniform(-1, 1))

        def __call__(self, x):
            # x * w + b 
            act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
            out = act.tanh() 
            return out

        def parameters(self):
            return self.w + [self.b]

    class Layer:
        def __init__(self, nin, nout):
            self. neurons = [Neuron(nin) for _ in range(nout)]

        def __call__(self, x):
            outs = [n(x) for n in self.neurons]
            return outs
        def parameters(self): 
            params = []
            params.extend(neuron.parameters() for neuron in self.neurons)
            return params

    class MLP:
        def __init__(self, nin, nouts):
            sz = [nin] + nouts
            self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x[0] if len(x) == 1 else x

        def parameters(self):
            return [param for layer in self.layers for neuron in layer.neurons for param in neuron.parameters()]
    return Layer, MLP, Neuron, random


@app.cell
def __():
    """
    Let's try and figure out why a single neuron is not apt for non-linear data. 

    The output of a neuron with inputs x1, x2 and weights w1, w2 and bias b is defined as,
        y = x1w1 + x2w2 + b

        The above equation defines a line in 2D plane.
        Hence each neuron slices the input space by a line, also known as its decision boundary. 

        Any point on one side of that line, the nueron classifies as activated or 1 and any point below it, the neuron classifies as 0 or weak.


    What is a decision boundary?
    A decision boundary of a neuron is a line above which the output of that neuron is >0 or 'activated' and below is < 0 or 'deactivated'.
    How do we compute the decision boundary?
    Let's take an example of x1 and x2 as inputs, w1 and w2 as weights and b as bias. 

    The neuron's output is y. 
    y is defined as,

        y = x1w1 + x2w2 +b
        This represents a line in 2D plane. 

    On the decision boundary, the output of the neuron is 0. 
    Hence, 
        x1w1 + x2w2 + b = 0 

    Rearranging terms, 
        x2 = -(w1/w2)x1 - (b / w2) 

        This represents a line with -(w1/w2) as its slope and - (b / w2) as its intercept (this point at which this line intersects x2 axis).
    """

    """
    Let's take an example of xor. The xor data is non-linearly scattered. 
    Let's see how a single neuron fails to classify a non-linearly scattered data.
    """
    return


@app.cell
def __(Neuron, np):
    # Assuming the Neuron and Value classes are defined as in the code provided above

    # Define XOR points and labels
    xor_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])

    # Create a single neuron with 2 inputs
    nn = Neuron(2)
    return labels, nn, xor_points


@app.cell
def __(labels, np, plt, w1, w2, xor_points):
    def plot_decision_boundary(weights, biases):
        # Retrieve weights and bias from the neuron
        # w1 = nn.w[0].data  # Weight for x1
        # w2 = nn.w[1].data  # Weight for x2
        # b = nn.b.data      # Bias

        all_x_vals = []
        all_y_vals = []

        for w, b in zip(weights, biases):
            x_vals = np.linspace(-0.5, 1.5, 100)
            if w[1] != 0:
                y_vals = - (w[0] / w[1]) * x_vals - (b / w[1])
            else:
                y_vals = np.full_like(x_vals, 10)

            all_x_vals.append(x_vals)
            all_y_vals.append(y_vals)

        # Calculate the decision boundary line based on neuron weights and bias
        x_vals = np.linspace(-0.5, 1.5, 100)
        if w2 != 0:
            y_vals = - (w1 / w2) * x_vals - (b / w2)
        else:
            # If w2 is zero, avoid division by zero and set y_vals to an arbitrary large value
            y_vals = np.full_like(x_vals, 10)

        # Plot XOR points
        plt.figure(figsize=(8, 6))
        plt.scatter(xor_points[labels == 0][:, 0], xor_points[labels == 0][:, 1], color='red', label='Class 0', s=100, edgecolor='k')
        plt.scatter(xor_points[labels == 1][:, 0], xor_points[labels == 1][:, 1], color='blue', label='Class 1', s=100, edgecolor='k')

        # Plot the decision boundary line
        for x, y in zip(all_x_vals, all_y_vals):
             plt.plot(x, y, 'k--', label="")

        # Set plot limits and labels
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.title("Decision Boundary Line Created by Single Neuron for XOR Data")
        plt.xlabel("Input x1")
        plt.ylabel("Input x2")
        plt.legend()
        plt.show()
    return (plot_decision_boundary,)


@app.cell
def __(nn, plot_decision_boundary):
    # Caputre original weights and bias of the neuron. So that when we update weights, we don't actually change the original neuron weights and biases
    weights = [[w.data for w in nn.w]]
    biases = [nn.b.data]

    plot_decision_boundary(weights, biases)
    return biases, weights


@app.cell
def __():
    """
    The boundary line above is created by randomly assigning weights and biases.
    """
    return


@app.cell
def __(nn, xor_points):
    #Obtain ys from neuron

    y = [] 

    for p in xor_points:
        y.append(nn(p))

    y3 = y[3]
    print(y3)
    return p, y, y3


@app.cell
def __(y3):
    # Do backprop to figure out how do weights and bias impact the output.
    y3.backward()
    return


@app.cell
def __(nn):
    print(f"grad of neuron of w1: {nn.w[0].grad}")
    print(f"grad of neuron of w2: {nn.w[1].grad}")
    print(f"grad of neuron of bias: {nn.b.grad}")
    return


@app.cell
def __():
    """
    From the original decision boundary, we can see that point(1,1) lies on the positive or 'activated' side of the output, but the actual output for point (1,1) should be 0 (1 xor 1 = 0). 
    Let's change the weights and bias gradually to accomodate output of y3, which should be close to 0. 
    Take a small h and start applying it to weights and biases.
    """
    return


@app.cell
def __(nn):
    weights_1 = [[w.data for w in nn.w]]
    biases_1 = [nn.b.data]
    h = 0.03
    w1_1, w2_1 = weights_1[0]
    b_1 = biases_1[0]
    for _ in range(10):
        w1_1 = w1_1 - h
        w2_1 = w2_1 - h
        b_1 = b_1 - h
        weights_1.append([w1_1, w2_1])
        biases_1.append(b_1)
    return b_1, biases_1, h, w1_1, w2_1, weights_1


@app.cell
def __(biases_1, plot_decision_boundary, weights_1):
    plot_decision_boundary(weights_1, biases_1)
    return


@app.cell
def __(biases_1, plot_decision_boundary, weights_1):
    plot_decision_boundary(weights_1[len(weights_1) - 1:], biases_1[len(biases_1) - 1:])
    return


@app.cell
def __():
    """
    As we can see, the last decision boundary after adjusting the weights, correctly classifies the point (1,1).
    But it still misclassifies (1, 0). Which is why, single neuron cannot correctly classify non-linear data.
    """
    return


@app.cell
def __():
    """ 
    Let's start defining multiple neurons, called a layer and see how using multiple neurons, 
    the data space is sliced into more complex regions.
    """
    return


@app.cell
def __(labels, np, plt, w1_1, w2_1, xor_points):
    def plot_decision_boundary_layer(weights, biases, colors):
        all_x_vals = []
        all_y_vals = []
        for w, b in zip(weights, biases):
            x_vals = np.linspace(-0.5, 1.5, 100)
            if w[1] != 0:
                y_vals = -(w[0] / w[1]) * x_vals - b / w[1]
            else:
                y_vals = np.full_like(x_vals, 10)
            all_x_vals.append(x_vals)
            all_y_vals.append(y_vals)
        x_vals = np.linspace(-0.5, 1.5, 100)
        if w2_1 != 0:
            y_vals = -(w1_1 / w2_1) * x_vals - b / w2_1
        else:
            y_vals = np.full_like(x_vals, 10)
        plt.figure(figsize=(8, 6))
        plt.scatter(xor_points[labels == 0][:, 0], xor_points[labels == 0][:, 1], color='red', label='Class 0', s=100, edgecolor='k')
        plt.scatter(xor_points[labels == 1][:, 0], xor_points[labels == 1][:, 1], color='blue', label='Class 1', s=100, edgecolor='k')
        for x, y, c in zip(all_x_vals, all_y_vals, colors):
            plt.plot(x, y, f'{c}--', label='')
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.title('Decision Boundary Line Created by Single Neuron for XOR Data')
        plt.xlabel('Input x1')
        plt.ylabel('Input x2')
        plt.legend()
        plt.show()
    return (plot_decision_boundary_layer,)


@app.cell
def __(Layer):
    layer = Layer(2, 10)
    return (layer,)


@app.cell
def __(layer, plot_decision_boundary_layer):
    weights_2 = [[n.w[0].data, n.w[1].data] for n in layer.neurons]
    biases_2 = [n.b.data for n in layer.neurons]
    colors = ['r', 'g']
    plot_decision_boundary_layer(weights_2, biases_2, colors)
    return biases_2, colors, weights_2


@app.cell
def __():
    # Define xor_inputs here
    xor_inputs = [[0,0], [0,1], [1,0], [1,1]]

    #This is the expected outputs for all the inputs above
    y_exp = [0, 1, 1, 0]
    return xor_inputs, y_exp


@app.cell
def __(layer):
    """
    Let's understand what exactly does 2 neurons in a layer means. 

    We have an input with 2 dimensions, x1 & x2. Both of these dimensions are fed to each neuron. 
    And the output of each neuron tells us whether it classifies the given (x1, x2) as 1 or zero. 
    In other words, it tells us that whether the neuron is active or inactive for the given input (x1, x2).

    The layer that we have defined as 2 neurons in it. 
    The output of layer tells us the output of each neuron for a given input (x1, x2)

    """

    y_pred = layer([0, 0])
    print(y_pred)

    """
    The above output tells us the output of each neuron for the input [0,0]
    The output of neuron 1 tells us that [0,0] activates it (or the output is close to 1). Obviously this is incorrect, as 0 xor 0 should be zero.
    The output of neuron 2 tells us that [0,0] deactivates it (or the output is close to 0). This is correct but coincidentally. 
    Since the weights and biases of both the neurons are randomly assigned, we are getting these values.

    """
    return (y_pred,)


@app.cell
def __(layer, xor_inputs):
    y_pred_neuron1 = []
    y_pred_neuron2 = []
    for input in xor_inputs:
        y_pred_1 = layer(input)
        y_pred_neuron1.append(y_pred_1[0])
        y_pred_neuron2.append(y_pred_1[1])
    (y_pred_neuron1, y_pred_neuron2)
    return input, y_pred_1, y_pred_neuron1, y_pred_neuron2


@app.cell
def __():
    """
    Let's try and modify the weights and biases of these neurons such that the outputs get closer and closer to actual expected outputs
    """
    return


@app.cell
def __():
    """
    Instead of manually adjusting the weights and biases this time, since we have double the number of weights and biases, we will try to do this
    in a more organised manner.

    Let's introduce the concept of a loss, here.

    Loss is defined as the distance between the output of a neuron from the actual expected output. It has a much broader explanation but this
    simple definition fits our current layer. 
    We will continue to evolve this definition as we go along.

    For each neuron, we will compute the loss. We will be using mean squared loss here, which defined as the sum of squared differences between
    neuron's outputs vs its inputs
    """
    return


@app.cell
def __():
    """
    Our goal is to update the weights and biases such that these indivisual losses of both the neurons reduce.
    We will do this one neuron at a time.
    """

    """
    Till now we have manually adjusted the values of weights and biases to move the output of a neuron closer to the expected output (in other words, 
    reduce the loss). 
    But now, we will do this with the following steps. 

    1. Define a small change value, h. 
    2. compute the outputs from the inputs (also known as a forward pass). 
    3. compute the loss
    4. Do backpropogation on the loss object, so that all the gradients of all the weights and biases are computed. 
    5. Update the weights and bias in order to reduce the overall loss
    6. Zero out the grads. Since we accumulate the gradients (+= in the Value class while computing gradients), we need to flush them so that 
    gradient computation starts from 0 for the new loss. 

    7. Iterate steps 2 to 6 and monitor the loss go down.
    """
    return


@app.cell
def __(Layer):
    layer_1 = Layer(2, 20)
    return (layer_1,)


@app.cell
def __(layer_1, plot_decision_boundary_layer):
    weights_3 = [[n.w[0].data, n.w[1].data] for n in layer_1.neurons]
    biases_3 = [n.b.data for n in layer_1.neurons]
    colors_1 = ['r', 'g']
    plot_decision_boundary_layer(weights_3, biases_3, colors_1)
    return biases_3, colors_1, weights_3


@app.cell
def __(layer_1, xor_inputs, y_exp):
    h_1 = 0.1
    min_loss = 100000
    apt_parameters = []
    loss = 0
    for k in range(500):
        loss = 0
        y_1 = [layer_1(input) for input in xor_inputs]
        for i in range(len(layer_1.neurons)):
            ypred = [d[i] for d in y_1]
            loss = loss + sum(((pred - exp) ** 2 for pred, exp in zip(ypred, y_exp)))
        loss = loss / len(layer_1.neurons)
        for neuron in layer_1.neurons:
            for p_1 in neuron.parameters():
                p_1.grad = 0
        loss.backward()
        for neuron in layer_1.neurons:
            for p_1 in neuron.parameters():
                p_1.data = p_1.data + -h_1 * p_1.grad
        if loss.data < min_loss:
            min_loss = loss.data
            params = []
            for neuron in layer_1.neurons:
                params.append([p.data for p in neuron.parameters()])
            apt_parameters = params
        if k % 100 == 0:
            print(f'pass: {k}, loss: {loss.data}')
    print(min_loss, apt_parameters)
    return (
        apt_parameters,
        h_1,
        i,
        k,
        loss,
        min_loss,
        neuron,
        p_1,
        params,
        y_1,
        ypred,
    )


@app.cell
def __(apt_parameters, plot_decision_boundary_layer):
    w = [weight[:len(weight) - 1] for weight in apt_parameters]
    b_2 = [weight[len(weight) - 1] for weight in apt_parameters]
    (w, b_2)
    colors_2 = ['r', 'g', 'b', 'k', 'y', 'c', 'm']
    num_neurons = len(w)
    colors_2 = colors_2[:num_neurons]
    plot_decision_boundary_layer(w, b_2, colors_2)
    return b_2, colors_2, num_neurons, w


@app.cell
def __(labels, layer_1, np, plt, xor_points):
    from matplotlib.colors import ListedColormap

    def plot_combined_decision_boundary(layer, xor_points, labels):
        x_min, x_max = (-0.5, 1.5)
        y_min, y_max = (-0.5, 1.5)
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        combined_output = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                point = [xx[i, j], yy[i, j]]
                neuron_outputs = [neuron(point).data for neuron in layer.neurons]
                combined_output[i, j] = 1 if sum((1 if o > 0 else 0 for o in neuron_outputs)) % 2 == 1 else 0
        plt.figure(figsize=(8, 6))
        plt.scatter(xor_points[labels == 0][:, 0], xor_points[labels == 0][:, 1], color='red', label='Class 0', s=100, edgecolor='k')
        plt.scatter(xor_points[labels == 1][:, 0], xor_points[labels == 1][:, 1], color='blue', label='Class 1', s=100, edgecolor='k')
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        plt.contourf(xx, yy, combined_output, cmap=cmap_light, alpha=0.3)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('Combined Decision Boundary for XOR')
        plt.xlabel('Input x1')
        plt.ylabel('Input x2')
        plt.legend()
        plt.show()
    plot_combined_decision_boundary(layer_1, xor_points, labels)
    return ListedColormap, plot_combined_decision_boundary


@app.cell
def __(Layer):
    layer_2 = Layer(2, 2)
    return (layer_2,)


@app.cell
def __(layer_2):
    print(layer_2.parameters())
    return


@app.cell
def __(mo):
    mo.md(r"""## Adding MLP to have multiple Layers now""")
    return


@app.cell
def __(ListedColormap, np, plt):
    def plot_combined_decision_boundary_mlp(mlp, xor_points, labels):
        # Generate a grid of points to evaluate
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        # For each point on the grid, calculate the output of the MLP
        combined_output = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                point = [xx[i, j], yy[i, j]]

                # Pass the point through the entire MLP to get the final output
                final_output = mlp(point)

                # If final_output is a list, retrieve the first element
                if isinstance(final_output, list):
                    final_output = final_output[0]

                # Use a threshold to classify the output: if final_output > 0, classify as 1; otherwise, classify as 0
                combined_output[i, j] = 1 if final_output.data > 0 else 0

        # Plot XOR points
        plt.figure(figsize=(8, 6))
        plt.scatter(xor_points[labels == 0][:, 0], xor_points[labels == 0][:, 1], color='red', label='Class 0', s=100, edgecolor='k')
        plt.scatter(xor_points[labels == 1][:, 0], xor_points[labels == 1][:, 1], color='blue', label='Class 1', s=100, edgecolor='k')

        # Plot the combined decision boundary
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        plt.contourf(xx, yy, combined_output, cmap=cmap_light, alpha=0.3)

        # Set plot limits and labels
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("Combined Decision Boundary for XOR using MLP")
        plt.xlabel("Input x1")
        plt.ylabel("Input x2")
        plt.legend()
        plt.show()
    return (plot_combined_decision_boundary_mlp,)


@app.cell
def __(MLP):
    n_1 = MLP(2, [2])
    return (n_1,)


@app.cell
def __(labels, n_1, plot_combined_decision_boundary, xor_points):
    plot_combined_decision_boundary(n_1.layers[0], xor_points, labels)
    return


@app.cell
def __(mo):
    mo.md(r"""#### We have added one more layer with a single neuron now. This single neuron will take the piecewise decision boundaries created by layer 1 as input and will apply linear transformation to that, along with a non linear activation function. Thus, making the combined decision boundary as non-linear.""")
    return


@app.cell
def __(
    Value,
    labels,
    n_1,
    plot_combined_decision_boundary,
    xor_inputs,
    xor_points,
):
    h_2 = 0.1
    for k_1 in range(100):
        ypred_1 = [n_1(input) for input in xor_inputs]
        loss_1 = sum([(y - Value(exp)) ** 2 for y, exp in zip(ypred_1, labels)])
        for p_2 in n_1.parameters():
            p_2.grad = 0
        loss_1.backward()
        for p_2 in n_1.parameters():
            p_2.data = p_2.data + -h_2 * p_2.grad
        if k_1 % 100 == 0:
            print(f'Pass: {k_1} | Loss: {loss_1.data}')
            plot_combined_decision_boundary(n_1, xor_points, labels)
    return h_2, k_1, loss_1, p_2, ypred_1


@app.cell
def __(labels, n_1, plot_combined_decision_boundary, xor_points):
    plot_combined_decision_boundary(n_1, xor_points, labels)
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
