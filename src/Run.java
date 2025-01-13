import monika.*;

// A simple program for testing the functionality of the neural network (XOR gate)
public class Run {
    static Network network;

    public static void main(String[] args) {
        network = new Network(2, 3, 1);
        train_xor();
        test_xor();
    }

    public static void train_xor() {
        System.out.println("Training started.");
        float[][][] trainingData = {{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}};
        float[] errors = network.train(trainingData, 10000, 2, 0.1f);
        for (int i = 0; i < errors.length; i++) {
            System.out.printf("Epoch " + (i + 1) + " mean error: %f", errors[i]);
            System.out.println();
        }
        System.out.println("Training finished.");
    }

    public static void test_xor() {
        test(0, 0); // 0
        test(0, 1); // 1
        test(1, 0); // 1
        test(1, 1); // 0
    }

    public static void test(float... inputs) {
        if (inputs.length != 2) {
            throw new IllegalArgumentException();
        }
        System.out.println();
        System.out.println("In: " + (int)inputs[0] + " " + (int)inputs[1]);
        System.out.printf("Out: %f", network.forward(inputs)[0]);
        System.out.println();
    }
}
