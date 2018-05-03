using System;
using MathNet.Numerics.LinearAlgebra;

namespace toy_nn_csharp
{
    public class NeuralNetwork
    {

        internal static class ActivationFunctions {
            public static ActivationFunction Sigmoid = new ActivationFunction
            {
                x = (x) => 1 / (1 + Math.Exp(-x)),
                y = (y) => y * (1 - y)
            };

            public static ActivationFunction TanH = new ActivationFunction
            {
                x = (x) => Math.Tanh(x),
                y = (y) => 1 - (y * y)
            };
        }
        internal class ActivationFunction
        {
            internal Func<double, double> x;
            internal Func<double, double> y;
        }

        int InputNodes { get; }
        int HiddenNodes { get;  }
        int OutputNodes { get; }

        Matrix<double> WeightsIh;
        Matrix<double> WeightsHo;
        Matrix<double> BiasH;
        Matrix<double> BiasO;

        ActivationFunction aFunction { get; set; } = ActivationFunctions.Sigmoid;


        private double learningRate = 0.1;

        public NeuralNetwork(int input, int hidden, int output) {
            InputNodes = input;
            HiddenNodes = hidden;
            OutputNodes = output;
            initMatrices();
        }

        private void initMatrices() {
            WeightsIh = Matrix<double>.Build.Random(HiddenNodes, InputNodes);
            WeightsHo = Matrix<double>.Build.Random(OutputNodes, HiddenNodes);
            BiasH = Matrix<double>.Build.Random(HiddenNodes, 1);
            BiasO = Matrix<double>.Build.Random(OutputNodes, 1);
        }


        public void train(double[] inputs, double[] targets) {
            var inputsMatrix = Matrix<double>.Build.DenseOfRowArrays(inputs);
            var targetsMatrix = Matrix<double>.Build.DenseOfRowArrays(targets);

            var hidden = (WeightsIh * inputsMatrix) + BiasH;
            hidden.Map(aFunction.x);
            var outputs = (WeightsHo * hidden) + BiasO;

            var outputErrors = targetsMatrix - outputs;
            var outputGradient = outputs.Clone().Map(aFunction.y) * outputErrors * learningRate;

            //hidden -> output delta
            var hoDelta = outputGradient * hidden.Transpose();
            WeightsHo += hoDelta;
            BiasO += outputGradient;

            var hiddenErrors = WeightsHo.Transpose() * outputErrors;
            var hiddenGradient = hidden.Clone().Map(aFunction.y) * hiddenErrors * learningRate;

            //input -> hidden delta
            var ihDelta = hiddenGradient * inputsMatrix.Transpose();
            WeightsIh += ihDelta;
            BiasH += hiddenGradient;
        }
    }
}
