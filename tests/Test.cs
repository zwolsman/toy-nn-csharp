using NUnit.Framework;
using System;
using ToyNN;
using MathNet.Numerics.LinearAlgebra;

namespace tests
{
    [TestFixture()]
    public class NeuralNetworkTests
    {
        [Test()]
        public void Create_Neural_Network()
        {
            var nn = new NeuralNetwork(2, 4, 1);
            Assert.IsNotNull(nn);
        }

        [Test]
        public void Train_Neural_Network()
        {
            var nn = new NeuralNetwork(2, 4, 1);
            var rng = new Random().NextDouble();
            var inputData = new[] { 1.0, 1.0 };
            var initialPredict = nn.Predict(inputData)[0];

            for (int i = 0; i < 1000; i++)
            {
                nn.Train(inputData, new[] { 1.0 });
            }
            var predictAfterTraining = nn.Predict(inputData)[0];

            Console.WriteLine("Initial prediction {0}", initialPredict);
            Console.WriteLine("After training: {0}", predictAfterTraining);
            Assert.IsTrue(predictAfterTraining > initialPredict);
            Assert.IsTrue(predictAfterTraining > 0.5);
            Assert.IsTrue(predictAfterTraining <= 1.0);
           
        }

        [Test]
        public void MatrixTest() {
            var m1 = Matrix<double>.Build.DenseOfColumnArrays(new[] { 1.0 });
            var m2 = Matrix<double>.Build.DenseOfColumnArrays(new[] { 0.9999 });
            var result = m1 - m2;
            Console.WriteLine(result);
        }
    }
}
