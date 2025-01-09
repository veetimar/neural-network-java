import monika.*;

public class Run {

    // XOR gate
    public static void main(String[] args) {
        Network network = new Network(2, 3, 1);
        float[][][] trainingData = {{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}};
        float[] errors = network.train(trainingData, 10000, 2, 0.1f);
        System.out.println("Training started.");
        for (int i = 0; i < errors.length; i++) {
            System.out.printf("Epoch " + (i + 1) + " mean error: %f", errors[i]);
            System.out.println();
        }
        System.out.println("Training finished.");
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