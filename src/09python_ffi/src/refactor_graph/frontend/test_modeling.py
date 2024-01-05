from refactor_graph.frontend.modeling import InfiniTensorModel, DTYPE
import numpy as np
import unittest


class TestModeling(unittest.TestCase):
    class AddOneModel(InfiniTensorModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __call__(self, input: str) -> str:
            super().__call__([input])
            bias = np.array(1).astype(np.int32)
            outputs = self.make_op("Add", {}, tuple([input, bias]))
            self.outputs.append(outputs[0])
            return outputs[0]

    def test_add_op(self):
        inputs = ["A"]
        model = self.AddOneModel()
        model(inputs[0])
        compiler = model.make_compiler({inputs[0]: (DTYPE.I32, [1, 2, 3])})
        executor = compiler.compile("cpu", "default", [])
        input = np.ones([1, 2, 3], dtype=np.int32)
        executor.set_input(0, input)
        executor.run()
        output = executor.get_output(0)
        self.assertTrue(np.array_equal(output, input + input))

    class AddTwoModel(InfiniTensorModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.addone1 = self.make_submodel(TestModeling.AddOneModel)
            self.addone2 = self.make_submodel(TestModeling.AddOneModel)

        def __call__(self, input: str):
            super().__call__([input])
            output = self.addone1(input)
            output = self.addone2(output)
            self.outputs.append(output)
            return output

    def test_submodule(self):
        inputs = ["A"]
        model = self.AddTwoModel()
        model(inputs[0])
        compiler = model.make_compiler({inputs[0]: (DTYPE.I32, [1, 2, 3])})
        executor = compiler.compile("cpu", "default", [])
        input = np.ones([1, 2, 3], dtype=np.int32)
        executor.set_input(0, input)
        executor.run()
        output = executor.get_output(0)
        self.assertTrue(np.array_equal(output, input + input + input))

    class AddThreeModel(InfiniTensorModel):
        def __init__(self, shape, **kwargs):
            super().__init__(**kwargs)
            self.addtwo = self.make_submodel(TestModeling.AddTwoModel)
            self.bias = self.parameter(np.zeros(shape, dtype=np.int32), "bias")

        def __call__(self, input: str):
            super().__call__([input])
            output = self.addtwo(input)
            output = self.make_op("Add", {}, tuple([output, self.bias]), ("output",))[0]
            self.outputs.append(output)
            return output
        
    def test_submodule_topo(self):
        inputs = ["A"]
        shape = [1, 2, 3]
        model = self.AddThreeModel(shape)
        model(inputs[0])
        add1 = "AddThreeModel/AddTwoModel/AddOneModel/Add"
        add2 = "AddThreeModel/AddTwoModel/AddOneModel_1/Add"
        add3 = "AddThreeModel/Add"
        self.assertTrue(add1 in model._nodes)
        self.assertTrue("A" == model._nodes[add1][0][0])
        self.assertTrue(add2 in model._nodes)
        self.assertTrue(model._nodes[add1][1][0] == model._nodes[add2][0][0])
        self.assertTrue(add3 in model._nodes)
        self.assertTrue(model._nodes[add2][1][0] == model._nodes[add3][0][0])
        self.assertTrue(model._nodes[add3][1][0] == "output")

    def test_submodule_with_param_loading(self):
        inputs = ["A"]
        shape = [1, 2, 3]
        model = self.AddThreeModel(shape)
        model(inputs[0])
        model.load_param({model.bias: np.ones(shape, dtype=np.int32)})
        compiler = model.make_compiler({inputs[0]: (DTYPE.I32, shape)})
        executor = compiler.compile("cpu", "default", [])
        input = np.ones(shape, dtype=np.int32)
        executor.set_input(0, input)
        executor.run()
        output = executor.get_output(0)
        self.assertTrue(np.array_equal(output, input + input + input + input))

if __name__ == "__main__":
    unittest.main()
