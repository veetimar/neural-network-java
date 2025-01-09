package monika;
import java.util.*;

public class Network {
    private final Layer[] layers;

    // Constructor for the whole neural network
    public Network(int... size) {
        if (size.length < 2) {
            throw new IllegalArgumentException("Cannot create a network smaller than 2 layers!");
        }
        this.layers = new Layer[size.length];
        this.layers[0] = new Layer(size[0]);
        for (int i = 1; i < size.length; i++) {
            if (i < size.length - 1) {
                this.layers[i] = new Layer(size[i], size[i - 1], false);
            } else {
                this.layers[i] = new Layer(size[i], size[i - 1], true);
            }
        }
        System.gc();
    }

    // Train the network using gradient descent (batch size -1 -> max batch size)
    public float[] train(float[][][] data, int epochs, int batchSize, float learningRate) {
        if (batchSize == -1) {
            batchSize = data.length;
        }
        if (batchSize > data.length || batchSize < 1) {
            throw new IllegalArgumentException("Illegal batch size!");
        }
        if (batchSize != data.length) {
            shuffle(data);
        }
        float[] errors = new float[epochs];
        for (int i = 0; i < epochs; i++) {
            float error = 0;
            for (int j = 0; j < data.length; j++) {
                if (j % batchSize == 0 && j > 0) {
                    update(learningRate, batchSize);
                }
                if (data[j].length != 2) {
                    throw new IllegalArgumentException("Illegal structure in training data!");
                }
                error += backward(data[j][0], data[j][1]);
            }
            update(learningRate, batchSize);
            errors[i] = error / data.length;
        }
        return errors;
    }

    // Forward inputs through the network and return outputs
    public float[] forward(float... inputs) {
        if (inputs.length != this.layers[0].length()) {
            throw new IllegalArgumentException("Illegal size for the inputs array!");
        }
        for (int i = 0; i < inputs.length; i++) {
            this.layers[0].neuron(i).setActivation(inputs[i]);
        }
        float[] outputs = new float[this.layers[this.layers.length - 1].length()];
        for (int i = 1; i < this.layers.length; i++) {
            boolean output = i == this.layers.length - 1;
            for (int j = 0; j < this.layers[i].length(); j++) {
                Neuron cn = this.layers[i].neuron(j);
                float value = 0;
                for (int k = 0; k < this.layers[i - 1].length(); k++) {
                    value += this.layers[i - 1].neuron(k).getActivation() * cn.getWeight(k);
                }
                value += cn.getBias();
                cn.setValue(value);
                if (!output) {
                    value = elu(value);
                    cn.setActivation(value);
                } else {
                    value = sigmoid(value);
                    cn.setActivation(value);
                    outputs[j] = value;
                }
            }
        }
        return outputs;
    }

    // Calculate and cache new weights and biases
    private float backward(float[] inputs, float[] expectedOutputs) {
        if (inputs.length != this.layers[0].length()) {
            throw new IllegalArgumentException("Illegal size for the inputs array!");
        } else if (expectedOutputs.length != this.layers[this.layers.length - 1].length()) {
            throw new IllegalArgumentException("Illegal size for the expected outputs array!");
        }
        resetGradients();
        float[] outputs = forward(inputs);
        float[] derivatives = squaredErrorDerivatives(outputs, expectedOutputs);
        for (int i = 0; i < this.layers[this.layers.length - 1].length(); i++) {
            Neuron n = this.layers[this.layers.length - 1].neuron(i);
            n.setGradient(sigmoidDerivative(n.getValue()) * derivatives[i]);
        }
        for (int i = this.layers.length - 1; i > 0; i--) {
            for (int j = 0; j < this.layers[i].length(); j++) {
                Neuron cn = this.layers[i].neuron(j);
                cn.addBias(cn.getGradient());
                for (int k = 0; k < this.layers[i - 1].length(); k++) {
                    Neuron pn = this.layers[i - 1].neuron(k);
                    cn.addWeight(k, pn.getActivation() * cn.getGradient());
                    if (i > 1) {
                        pn.setGradient(pn.getGradient() + eluDerivative(pn.getValue()) * cn.getWeight(k) * cn.getGradient());
                    }
                }
            }
        }
        return meanSquaredError(outputs, expectedOutputs);
    }

    private void resetGradients() {
        for (int i = 1; i < this.layers.length - 1; i++) {
            for (int j = 0; j < this.layers[i].length(); j++) {
                this.layers[i].neuron(j).setGradient(0);
            }
        }
    }

    private float meanSquaredError(float[] outputs, float[] expectedOutputs) {
        return squaredError(outputs, expectedOutputs) / outputs.length;
    }

    private float squaredError(float[] outputs, float[] expectedOutputs) {
        if (outputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("The outputs and expected outputs arrays differ in length!");
        }
        float error = 0;
        for (int i = 0; i < outputs.length; i++) {
            error += Math.pow(outputs[i] - expectedOutputs[i], 2);
        }
        return error;
    }

    private float[] squaredErrorDerivatives(float[] outputs, float[] expectedOutputs) {
        if (outputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("The outputs and expected outputs arrays differ in length!");
        }
        float[] derivatives = new float[outputs.length];
        for (int i = 0; i < derivatives.length; i++) {
            derivatives[i] = 2 * (outputs[i] - expectedOutputs[i]);
        }
        return derivatives;
    }

    private void shuffle(float[][][] array) {
        List<float[][]> list = Arrays.asList(array);
        Collections.shuffle(list);
    }

    private float sigmoid(float x) {
        return (float)(1 / (1 + Math.exp(-x)));
    }

    private float sigmoidDerivative(float x) {
        return (float)(Math.exp(x) / Math.pow(1 + Math.exp(x), 2));
    }

    private float elu(float x) {
        if (x >= 0) {
            return x;
        } else {
            return (float)(Math.exp(x) - 1);
        }
    }

    private float eluDerivative(float x) {
        if (x >= 0) {
            return 1;
        } else {
            return (float)Math.exp(x);
        }
    }

    private void update(float learningRate, int batchSize) {
        for (int i = 1; i < this.layers.length; i++) {
            this.layers[i].update(learningRate, batchSize);
        }
    }

    public Layer layer(int index) {
        return this.layers[index];
    }

    public int length() {
        return this.layers.length;
    }

    public String toString() {
        String s = new String();
        for (int i = 0; i < this.layers.length; i++) {
            s += this.layers[i];
            if (i < this.layers.length - 1) {
                s += "\n";
            }
        }
        return s;
    }
}

class Layer {
    private final Neuron[] neurons;

    // Constructor for input layer
    public Layer(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("Cannot create a layer smaller than 1 neuron!");
        }
        this.neurons = new Neuron[size];
        for (int i = 0; i < this.neurons.length; i++) {
            this.neurons[i] = new Neuron();
        }
    }

    // Constructor for other layers
    public Layer(int size, int previousSize, boolean output) {
        if (size < 1) {
            throw new IllegalArgumentException("Cannot create a layer smaller than 1 neuron!");
        } else if (previousSize < 1) {
            throw new IllegalArgumentException("Illegal size for the previous layer!");
        }
        this.neurons = new Neuron[size];
        if (!output) {
            for (int i = 0; i < this.neurons.length; i++) {
                this.neurons[i] = new Neuron(he(previousSize), 0);
            }
        } else {
            for (int i = 0; i < this.neurons.length; i++) {
                this.neurons[i] = new Neuron(xavier(previousSize), 0);
            }
        }
    }

    // Elu weight initialization
    private float[] he(int size) {
        float[] weights = new float[size];
        Random r = new Random();
        float stddev = (float)Math.sqrt(2.0 / size);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float)r.nextGaussian(0, stddev);
        }
        return weights;
    }

    // Sigmoid weight initialization
    private float[] xavier(int size) {
        float[] weights = new float[size];
        Random r = new Random();
        float min = (float)(-1.0 / Math.sqrt(size));
        float max = (float)(1.0 / Math.sqrt(size));
        for (int j = 0; j < weights.length; j++) {
            weights[j] = r.nextFloat(min, max);
        }
        return weights;
    }

    public void update(float learningRate, int batchSize) {
        for (int i = 0; i < this.neurons.length; i++) {
            this.neurons[i].update(learningRate, batchSize);
        }
    }

    public Neuron neuron(int index) {
        return this.neurons[index];
    }

    public int length() {
        return this.neurons.length;
    }

    public String toString() {
        String s = new String();
        for (int i = 0; i < this.neurons.length; i++) {
            s += this.neurons[i];
            if (i < this.neurons.length - 1) {
                s += " ";
            }
        }
        return s;
    }
}

class Neuron {
    private float value;
    private float activation;
    private float bias;
    private float deltaBias;
    private final float[] weights;
    private final float[] deltaWeights;
    private float gradient;

    // Constructor for input neuron
    public Neuron() {
        this.weights = null;
        this.deltaWeights = null;
    }

    // Constructor for other neurons
    public Neuron(float[] weights, float bias) {
        this.bias = bias;
        this.weights = new float[weights.length];
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = weights[i];
        }
        this.deltaWeights = new float[this.weights.length];
    }

    public void update(float learningRate, int batchSize) {
        learningRate /= batchSize;
        this.bias -= learningRate * this.deltaBias;
        this.deltaBias = 0;
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] -= learningRate * this.deltaWeights[i];
            this.deltaWeights[i] = 0;
        }
        this.gradient = 0;
    }

    public void setValue(float value) {
        this.value = value;
    }

    public void setActivation(float activation) {
        this.activation = activation;
    }

    public void addBias(float bias) {
        this.deltaBias += bias;
    }

    public void addWeight(int index, float weight) {
        this.deltaWeights[index] += weight;
    }

    public void setGradient(float gradient) {
        this.gradient = gradient;
    }

    public float getValue() {
        return this.value;
    }

    public float getActivation() {
        return this.activation;
    }

    public float getBias() {
        return this.bias;
    }

    public float getWeight(int index) {
        return this.weights[index];
    }

    public float getGradient() {
        return this.gradient;
    }

    public String toString() {
        return String.valueOf(this.activation);
    }
}
