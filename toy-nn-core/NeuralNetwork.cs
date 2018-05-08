﻿using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace toynncore
{
    public class NeuralNetwork
    {
        private const double LEARNING_RATE = 0.1;

        private readonly Random _random = new Random();
        private readonly Matrix<double>[] _weights;
        private readonly Matrix<double>[] _biases;
        private readonly int[] _layers;

        public NeuralNetwork(IEnumerable<int> layers)
        {
            _layers = layers.ToArray();

            if (_layers.Length < 3)
                throw new ArgumentException($"Layers should be greater than 3. Got {_layers.Length}");
            
            _weights = new Matrix<double>[_layers.Length - 1];
            _biases = new Matrix<double>[_layers.Length - 1];

            for (var i = 1; i < _layers.Length; i++)
            {
                _weights[i - 1] = Matrix<double>.Build.Dense(_layers[i], _layers[i - 1], Rng);
                _biases[i - 1] = Matrix<double>.Build.Dense(_layers[i], 1, Rng);
            }
        }

        private int InputNodes => _layers.First();
        
        private int OutputNodes => _layers.Last();

        private ActivationFunction _activationFunction { get; } = ActivationFunctions.Sigmoid;

        public double[] Predict(params double[] inputsArray)
        {
            if (inputsArray.Length != InputNodes)
                throw new ArgumentException($"Inputs should be the same amount as the input nodes. Expected {InputNodes} got {inputsArray.Length}");
            
            var outputs = Matrix<double>.Build.DenseOfColumnArrays(inputsArray);
            for (var i = 0; i < _layers.Length - 1; i++)
            {
                var weight = _weights[i];
                var bias = _biases[i];
                var layer = ((weight * outputs) + bias).Map(_activationFunction.func);
                outputs = layer;
            }
            return outputs.Column(0).ToArray();
        }

        public void Train(double[] inputsArray, double[] targetsArray)
        {
            if (inputsArray.Length != InputNodes)
                throw new ArgumentException($"Inputs should be the same amount as the input nodes. Expected {InputNodes} got {inputsArray.Length}");

            if (targetsArray.Length != OutputNodes)
                throw new ArgumentException($"Targets should be the same amount as the output nodes. Expected {OutputNodes} got {targetsArray.Length}");

            var inputs = Matrix<double>.Build.DenseOfColumnArrays(inputsArray);
            var targets = Matrix<double>.Build.DenseOfColumnArrays(targetsArray);
            var layers = new Matrix<double>[_layers.Length];
            layers[0] = inputs;
            //Setup layers
            for (var i = 1; i < _layers.Length; i++)
            {
                var weight = _weights[i - 1];
                var bias = _biases[i - 1];
                var layer = (weight * inputs) + bias;
                layer.Map(_activationFunction.func, layer);

                layers[i] = layer;
                inputs = layer;
            }

            for (var i = _layers.Length - 1; i > 0; i--)
            {
                var errors = targets - layers[i];

                var gradients = layers[i].Map(_activationFunction.fund).PointwiseMultiply(errors) * LEARNING_RATE;
                var deltas = gradients * layers[i - 1].Transpose();

                _biases[i - 1] += gradients;
                _weights[i - 1] += deltas;

                var prevErrors = _weights[i - 1].Transpose() * errors;
                targets = prevErrors + layers[i - 1];
            }
        }

        private double Rng(int arg1, int arg2)
        {
            return (_random.NextDouble() * 2) - 1;
        }

        static class ActivationFunctions
        {
            public static readonly ActivationFunction Sigmoid = new ActivationFunction(
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