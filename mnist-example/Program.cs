using System;
using System.IO;
using System.Linq;
using toynncore;

namespace mnist_example
{
    class Program
    {

        const int Cycli = 50;

        static Random _random = new Random();

        static void Main(string[] args)
        {
            var trainLabels = File.ReadAllBytes("data/train-labels.idx1-ubyte").Skip(sizeof(int) * 2).ToArray();
            var trainImages = File.ReadAllBytes("data/train-images.idx3-ubyte").Skip(sizeof(int)).ToArray();

            var totalImages = ReadInt(trainImages);
            var rows = ReadInt(trainImages, sizeof(int));
            var cols = ReadInt(trainImages, sizeof(int) * 2);

            if (totalImages != trainLabels.Length)
                throw new FormatException("The training data doesn't match the training labels");
            Console.WriteLine($"Number of images: {totalImages}");

            var trainingData = new Tuple<double[], double[]>[totalImages];

            var inputs = rows * cols;

            for (int i = 0; i < totalImages; i++) {
                var label = trainLabels[i];
                var data = trainImages.Skip(inputs * i).Take(inputs).Select(b => b / 255.0).ToArray();
                trainingData[i] = new Tuple<double[], double[]>(CreateTargets(label), data);
            }


            var tindex = _random.Next(totalImages);
            var target = trainLabels[tindex];
            Console.WriteLine("Target: " + target);
            //Cleanup?
            trainLabels = null;
            trainImages = null;
            Console.Write("Guess: -");
            var nn = new NeuralNetwork(inputs, 128, 32, 16, 10);
            while(true) {
                for (int i = 0; i < Cycli; i++) {
                    var data = trainingData[_random.Next(totalImages)];
                    nn.Train(data.Item2, data.Item1);
                }

                var result = nn.Predict(trainingData[tindex].Item2).ToList();
                var digit = result.IndexOf(result.Max());
                Console.CursorLeft -= 1;
                Console.Write(digit);
               
            }
        }

        private static double[] CreateTargets(int target) {
            var targets = new double[10];
            targets[target] = 1.0;
            return targets;
        }

        private static int ReadInt(byte[] input, int offset = 0) {

            var raw = input.Skip(offset).Take(sizeof(int));
            if (BitConverter.IsLittleEndian)
                raw = raw.Reverse();
            return BitConverter.ToInt32(raw.ToArray(), 0);
        }
    }
}
