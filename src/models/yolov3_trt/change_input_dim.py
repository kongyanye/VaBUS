import onnx


def change_input_dim(model):

    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = 1


model = onnx.load('yolov3-416.onnx')
change_input_dim(model)
onnx.save(model, 'yolov3-416.onnx.b1')
