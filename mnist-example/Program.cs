using System;
using System.IO;
using System.Linq;

namespace mnist_example
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var trainLabels = File.ReadAllBytes("data/train-labels.idx1-ubyte").Skip(sizeof(int) * 2).ToArray();
            var trainImages = File.ReadAllBytes("data/train-images.idx3-ubyte").Skip(sizeof(int)).ToArray();


            var totalImages = ReadInt(trainImages);
            var rows = ReadInt(trainImages, sizeof(int));
            var cols = ReadInt(trainImages, sizeof(int) * 2);

            if (totalImages != trainLabels.Length)
                throw new FormatException("The training data doesn't match the training labels");
            Console.WriteLine($"Number of images: {totalImages}");
        }

        private static int ReadInt(byte[] input, int offset = 0) {

            var raw = input.Skip(offset).Take(sizeof(int));
            if (BitConverter.IsLittleEndian)
                raw = raw.Reverse();
            return BitConverter.ToInt32(raw.ToArray(), 0);
        }
    }
}
