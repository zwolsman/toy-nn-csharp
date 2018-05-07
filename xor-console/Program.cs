using System;
using Console = Colorful.Console;
using System.Drawing;
using toy_nn_core.toynncore;
using System.Diagnostics;

namespace xor_console
{
    class TrainingData {
        public double[] inputs { get; set; }
        public double[] targets { get; set; }
    }

    class Program
    {

        const int Rows = 10;
        const int Cols = 10;


        static TrainingData[] trainings = new[] {
                new TrainingData {
                    inputs = new[] {0.0, 0.0},
                    targets = new[] {0.0}
                },
                new TrainingData {
                    inputs = new[] {1.0, 0.0},
                    targets = new[] {1.0}
                },
                new TrainingData {
                    inputs = new[] {0.0, 1.0},
                    targets = new[] {1.0}
                },
                new TrainingData {
                    inputs = new[] {1.0, 1.0},
                    targets = new[] {0.0}
                },
            };

        static void Main(string[] args)
        {
            var nn = new NeuralNetwork(2, 4, 1);
            var rng = new Random();
            while(true)
            {
                for (int i = 0; i < 10; i++) {
                    var training = trainings[rng.Next(trainings.Length)];
                    nn.Train(training.inputs, training.targets);
                }
               
                for (int x = 0; x <= Rows; x++)
                {
                    for (int y = 0; y <= Cols; y++)
                    {
                        //Debug.WriteLine(x/Rows);
                        double x1 = x / Rows;
                        double x2 = y / Cols;
                        var inputs = new[] { x1, x2 };
                        var outputs = nn.Predict(inputs);
                        //Debug.WriteLine($"Inputs: {String.Join(",",inputs)}");
                        //Debug.WriteLine($"Outputs: {String.Join(",", outputs)}");
                        var c = (int)(outputs[0] * 255);
                        Console.BackgroundColor = Color.FromArgb(c);
                        Console.Write(String.Format("{0,3:D}", c) + " ", Color.Purple);
                        //Console.Write("■", Color.FromArgb(c));
                    }
                    Console.WriteLine();
                }
                Console.CursorTop = 0;
                Console.CursorLeft = 0;
            }
        }
    }
}
