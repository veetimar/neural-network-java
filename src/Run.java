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
        System.out.println();
        System.out.println("In: " + "0  0");
        System.out.printf("Out: %f", network.forward(0, 0)[0]);
        System.out.println();
        System.out.println();
        System.out.println("In: " + "0  1");
        System.out.printf("Out: %f", network.forward(0, 1)[0]);
        System.out.println();
        System.out.println();
        System.out.println("In: " + "1  0");
        System.out.printf("Out: %f", network.forward(1, 0)[0]);
        System.out.println();
        System.out.println();
        System.out.println("In: " + "1  1");
        System.out.printf("Out: %f", network.forward(1, 1)[0]);
        System.out.println();
    }
}