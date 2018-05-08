using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace toynncore
{
    public class NeuralNetwork
    {
        private readonly double learningRate = 0.1;
        private readonly Random random = new Random();
        private Matrix<double>[] _weights;
        private Matrix<double>[] _biases;

        private int[] _layers;
        public NeuralNetwork(params int[] layers)
        {
            if (layers.Length < 3)
            {
                throw new ArgumentException($"Layers should be greater than 3. Got {layers.Length}");
            }
            _layers = layers;
            _weights = new Matrix<double>[layers.Length - 1];
            _biases = new Matrix<double>[layers.Length - 1];
            for (var i = 1; i < layers.Length; i++)
            {
                _weights[i - 1] = Matrix<double>.Build.Dense(layers[i], layers[i - 1], Rng);
                _biases[i - 1] = Matrix<double>.Build.Dense(layers[i], 1, Rng);
            }
        }

        public int InputNodes
        {
            get
            {
                return _layers[0];
            }
        }

        public int[] HiddenNodes
        {
            get
            {
                return _layers.Skip(1).Take(_layers.Length - 2).ToArray();
            }
        }

        public int OutputNodes
        {
            get
            {
                return _layers[_layers.Length - 1];
            }
        }

        private ActivationFunction _activationFunction { get; } = ActivationFunctions.Sigmoid;

        public double[] Predict(params double[] inputs)
        {
            if (inputs.Length != InputNodes)
                throw new ArgumentException($"Inputs should be the same amount as the input nodes. Expected {InputNodes} got {inputs.Length}");

            var inputsMatrix = Matrix<double>.Build.DenseOfColumnArrays(inputs);
            var hidden = ((WeightsIh * inputsMatrix) + BiasH).Map(_activationFunction.func);

            var outputs = ((WeightsHo * hidden) + BiasO).Map(_activationFunction.func);
            return outputs.Column(0).ToArray();
        }

        public void Train(double[] inputsArray, double[] targetsArray)
        {
            if (inputsArray.Length != InputNodes)
                throw new ArgumentException($"Inputs should be the same amount as the input nodes. Expected {InputNodes} got {inputs.Length}");

            if (targetsArray.Length != OutputNodes)
                throw new ArgumentException($"Targets should be the same amount as the output nodes. Expected {OutputNodes} got {targets.Length}");


            var inputs = Matrix<double>.Build.DenseOfColumnArrays(inputsArray);
            var targets = Matrix<double>.Build.DenseOfColumnArrays(targetsArray);
            var layers = new Matrix<double>[_layers.Length];
            layers[0] = inputs;
            layers[_layers.Length - 1] = targets;

            //Setup layers
            for (int i = 1; i < _layers.Length; i++) {
                var weight = _weights[i - 1];
                var bias = _biases[i - 1];
                var layer = (weight * inputs) + bias;
                layer.Map(_activationFunction.func, layer);
                layers[i] = layer;
            }


            //var hidden = ((WeightsIh * inputsMatrix) + BiasH).Map(aFunction.x);

            //var outputs = ((WeightsHo * hidden) + BiasO).Map(aFunction.x);

            //var outputErrors = targetsMatrix - outputs;
            //var outputGradient = outputs.Map(aFunction.y) * outputErrors * learningRate;

            ////hidden -> output delta
            //var hoDelta = outputGradient * hidden.Transpose();
            //WeightsHo += hoDelta;
            //BiasO += outputGradient;

            //var hiddenErrors = WeightsHo.Transpose() * outputErrors;

            //var hiddenGradient = hidden.Map(aFunction.y).PointwiseMultiply(hiddenErrors) * learningRate;

            ////input -> hidden delta
            //var ihDelta = hiddenGradient * inputsMatrix.Transpose();
            //WeightsIh += ihDelta;
            //BiasH += hiddenGradient;

            //return outputErrors[0, 0];
        }

        private double Rng(int arg1, int arg2)
        {
            return (random.NextDouble() * 2) - 1;
        }

        static class ActivationFunctions
        {
            public static ActivationFunction Sigmoid = new ActivationFunction(
                x => 1 / (1 + Math.Exp(-x)),
                y => y * (1 - y)
            );

            public static ActivationFunction TanH = new ActivationFunction(
                Math.Tanh,
                y => 1 - (y * y)
            );
        }

        class ActivationFunction
        {
            public Func<double, double> func { get; }
            public Func<double, double> fund { get; }

            public ActivationFunction(Func<double, double> x, Func<double, double> y)
            {
                this.func = x;
                this.fund = y;
            }
        }
    }
}