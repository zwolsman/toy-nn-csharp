using System;
using System.Diagnostics;
using System.Drawing;
using toynncore;
using Console = Colorful.Console;

namespace xor_console
{
    internal class TrainingData
    {
        public double[] Inputs { get; set; }

        public double[] Targets { get; set; }
    }

    internal class Program
    {
        private static readonly TrainingData[] _trainings =
        {
            new TrainingData
            {
                Inputs = new[] {0.0, 0.0},
                Targets = new[] {0.0}
            },
            new TrainingData
            {
                Inputs = new[] {1.0, 0.0},
                Targets = new[] {1.0}
            },
            new TrainingData
            {
                Inputs = new[] {0.0, 1.0},
                Targets = new[] {1.0}
            },
            new TrainingData
            {
                Inputs = new[] {1.0, 1.0},
                Targets = new[] {0.0}
            }
        };

        private static void Main(string[] args)
        {
            var nn = new NeuralNetwork(2, 4, 1);
            var rng = new Random();


            var rows = 10.0;
            var cols = 10.0;

            while (true)
            {
                for (var i = 0; i < 10; i++)
                {
                    var training = _trainings[rng.Next(_trainings.Length)];
                    var gradient = nn.Train(training.Inputs, training.Targets);

                    Debug.WriteLine(gradient);
                }

                for (var x = 0; x <= rows; x++)
                {
                    for (var y = 0; y <= cols; y++)
                    {
                        var x1 = Math.Round(x / rows);
                        var x2 = Math.Round(y / cols);
                        var outputs = nn.Predict(x1, x2);
                        var c = (int) (outputs[0] * 255);
                        Console.Write($"{c,3:D} ", Color.FromArgb(c, Color.White));
                    }
                    Console.WriteLine();
                }

                Console.CursorLeft = 0;
                Console.CursorTop = 0;
            }
        }
    }
}