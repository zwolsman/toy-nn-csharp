using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using toynncore;
using System.Linq;
using Console = Colorful.Console;

namespace xor_console
{
    internal class Program
    {
        private static readonly IReadOnlyList<TrainingData> _trainings = new[]
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
            var nn = new NeuralNetwork(2, 8, 4,2, 1);
            var rng = new Random();

            var resetTop = Console.CursorTop;
            var rows = 20.0;
            var cols = 20.0;

            while (true)
            {
                var cost = 0.0;
                var cycli = 50;
                for (var i = 0; i < cycli; i++)
                {
                    var training = _trainings[rng.Next(_trainings.Count)];
                    nn.Train(training.Inputs, training.Targets);
                    var result = nn.Predict(training.Inputs)[0];
                    var expected = training.Targets[0];
                    cost += Math.Abs(expected - result);
                }
                cost /= cycli;
                Debug.WriteLine("Cost: {0}", cost);

                for (var x = 0; x <= cols; x++)
                {
                    for (var y = 0; y <= rows; y++)
                    {
                        var x1 = Math.Round(x / rows);
                        var x2 = Math.Round(y / cols);
                        var outputs = nn.Predict(x1, x2);
                        var c = (int)(outputs[0] * 255);
                        Console.Write("█", Color.FromArgb(c, c, c));
                    }
                    Console.WriteLine();
                }

                Console.CursorLeft = 0;
                Console.CursorTop = resetTop;
            }
        }

        private class TrainingData
        {
            public double[] Inputs { get; set; }

            public double[] Targets { get; set; }
        }
    }
}