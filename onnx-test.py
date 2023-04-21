""" silliness using python to create an onnx model

from <https://onnx.ai/onnx/intro/python.html>

"""

from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# inputs
X_NAME = "Drews_Height"
Y_NAME = "Drews_Squat"
ADDITIONAL_WIN = "Magic"
RESULT_NODE = "Partial_Baguette"

# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
X = make_tensor_value_info(X_NAME, TensorProto.INT8, [None, None])
A = make_tensor_value_info(Y_NAME, TensorProto.FLOAT, [None, None])
B = make_tensor_value_info(ADDITIONAL_WIN, TensorProto.FLOAT, [None, None])

# outputs, the shape is left undefined

Y = make_tensor_value_info(RESULT_NODE, TensorProto.FLOAT, [None])

# nodes

# It creates a node defined by the operator type MatMul,
# 'X', 'A' are the inputs of the node, 'XA' the output.
node1 = make_node('MatMul', [X_NAME, Y_NAME], ['XA'])
node2 = make_node('Add', ['XA', ADDITIONAL_WIN], [RESULT_NODE])

# from nodes to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.

graph = make_graph([node1, node2],  # nodes
                    'lr',  # a name
                    [X, A, B],  # inputs
                    [Y])  # outputs

# onnx graph
# there is no metadata in this case.
onnx_model = make_model(graph)

# Let's check the model is consistent,
# this function is described in section
# Checker and Shape Inference.
check_model(onnx_model)

# The serialization
filename = "squat_model.onnx"
with open(filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(onnx_model)
print(f"wrote to {filename}")