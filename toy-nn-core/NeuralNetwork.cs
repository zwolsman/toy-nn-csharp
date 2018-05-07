using System;

namespace toy_nn_core
{
    using System;
    using MathNet.Numerics.LinearAlgebra;

    namespace toynncore
    {
        public class NeuralNetwork
        {
            internal static class ActivationFunctions
            {
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

            public int InputNodes { get; }
            public int HiddenNodes { get; }
            public int OutputNodes { get; }

            Matrix<double> WeightsIh;
            Matrix<double> WeightsHo;
            Matrix<double> BiasH;
            Matrix<double> BiasO;
            Random random = new Random();
            double rng(int arg1, int arg2) => random.NextDouble() * 2 - 1;

            ActivationFunction aFunction { get; set; } = ActivationFunctions.Sigmoid;


            double learningRate = 0.1;

            public NeuralNetwork(int input, int hidden, int output)
            {
                InputNodes = input;
                HiddenNodes = hidden;
                OutputNodes = output;
                initMatrices();
            }

            private void initMatrices()
            {
                WeightsIh = Matrix<double>.Build.Dense(HiddenNodes, InputNodes, rng);
                WeightsHo = Matrix<double>.Build.Dense(OutputNodes, HiddenNodes, rng);
                BiasH = Matrix<double>.Build.Dense(HiddenNodes, 1, rng);
                BiasO = Matrix<double>.Build.Dense(OutputNodes, 1, rng);
            }

            public void PrintMatrices()
            {
                Console.WriteLine("Weights input -> hidden");
                Console.WriteLine(WeightsIh);
                Console.WriteLine("Weights hidden -> output");
                Console.WriteLine(WeightsHo);
                Console.WriteLine("Bias hidden");
                Console.WriteLine(BiasH);
                Console.WriteLine("Bias output");
                Console.WriteLine(BiasO);
            }

            public double[] Predict(params double[] inputs)
            {
                if (inputs.Length != InputNodes)
                {
                    throw new ArgumentException($"Inputs should be the same amount as the input nodes. Expected {InputNodes} got {inputs.Length}");
                }
                var inputsMatrix = Matrix<double>.Build.DenseOfColumnArrays(inputs);
                var hidden = ((WeightsIh * inputsMatrix) + BiasH).Clone().Map(aFunction.x);

                var outputs = ((WeightsHo * hidden) + BiasO).Map(aFunction.x);
                return outputs.Column(0).ToArray();
            }

            public void Train(double[] inputs, double[] targets)
            {

                if (inputs.Length != InputNodes)
                {
                    throw new ArgumentException($"Inputs should be the same amount as the input nodes. Expected {InputNodes} got {inputs.Length}");
                }
                if (targets.Length != OutputNodes)
                {
                    throw new ArgumentException($"Targets should be the same amount as the output nodes. Expected {OutputNodes} got {targets.Length}");
                }

                var inputsMatrix = Matrix<double>.Build.DenseOfColumnArrays(inputs);
                var targetsMatrix = Matrix<double>.Build.DenseOfColumnArrays(targets);


                var hidden = ((WeightsIh * inputsMatrix) + BiasH).Clone().Map(aFunction.x);

                var outputs = ((WeightsHo * hidden) + BiasO).Map(aFunction.x);

                var outputErrors = targetsMatrix - outputs;
                var outputGradient = (outputs.Clone().Map(aFunction.y) * outputErrors) * learningRate;

                //hidden -> output delta
                var hoDelta = outputGradient * hidden.Transpose();
                WeightsHo += hoDelta;
                BiasO += outputGradient;

                var hiddenErrors = WeightsHo.Transpose() * outputErrors;

                var hiddenGradient = hidden.PointwiseMultiply(hiddenErrors) * learningRate;

                //input -> hidden delta
                var ihDelta = hiddenGradient * inputsMatrix.Transpose();
                WeightsIh += ihDelta;
                BiasH += hiddenGradient;
            }
        }
    }

}
