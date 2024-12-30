package monika;
import java.util.*;

public class Network {
    private final Layer[] layers;

    public Network(int... size) {
        if (size.length < 2) {
            throw new IllegalArgumentException("Cannot create a neural network smaller than 2 layers!");
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

    public float[][] train(float[][][] data, int times, float learningRate) {
        shuffle(data);
        float[][] errors = new float[times][data.length];
        for (int i = 0; i < times; i++) {
            for (int j = 0; j < data.length; j++) {
                if (data[j].length != 2) {
                    throw new IllegalArgumentException("Illegal structure in training data!");
                }
                errors[i][j] = backward(data[j][0], data[j][1], learningRate);
            }
        }
        return errors;
    }

    public float[] forward(float[] inputs) {
        if (inputs.length != layers[0].length()) {
            throw new IllegalArgumentException("Illegal size for the inputs array!");
        }
        for (int i = 0; i < inputs.length; i++) {
            layers[0].neuron(i).setActivation(inputs[i]);
        }
        float[] outputs = new float[layers[layers.length - 1].length()];
        for (int i = 1; i < layers.length; i++) {
            boolean output = i == layers.length - 1;
            for (int j = 0; j < layers[i].length(); j++) {
                Neuron cn = layers[i].neuron(j);
                float value = 0;
                for (int k = 0; k < layers[i - 1].length(); k++) {
                    value += layers[i - 1].neuron(k).getActivation() * cn.getWeight(k);
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

    public float backward(float[] inputs, float[] expectedOutputs, float learningRate) {
        if (inputs.length != layers[0].length()) {
            throw new IllegalArgumentException("Illegal size for the inputs array!");
        } else if (expectedOutputs.length != layers[layers.length - 1].length()) {
            throw new IllegalArgumentException("Illegal size for the expected outputs array!");
        }
        float[] outputs = forward(inputs);
        float error = meanSquaredError(outputs, expectedOutputs);
        float[] errorpd = squaredErrorPD(outputs, expectedOutputs);
        for (int i = 0; i < layers[layers.length - 1].length(); i++) {
            layers[layers.length - 1].neuron(i).setGradient(errorpd[i]);
        }
        for (int i = layers.length - 1; i > 0; i-- ) {
            boolean output = i == layers.length - 1;
            for (int j = 0; j < layers[i].length(); j++) {
                Neuron cn = layers[i].neuron(j);
                float gradient;
                if (!output) {
                    gradient = cn.getGradient() * eluD(cn.getValue());
                } else {
                    gradient = cn.getGradient() * sigmoidD(cn.getValue());
                }
                cn.setBias(cn.getBias() - learningRate * gradient);
                for (int k = 0; k < layers[i - 1].length(); k++) {
                    cn.setWeight(k, cn.getWeight(k) - learningRate * layers[i - 1].neuron(k).getActivation() * gradient);
                    if (i > 1) {
                        layers[i - 1].neuron(k).setGradient(layers[i - 1].neuron(k).getGradient() + cn.getWeight(k) * gradient);
                    }
                }
            }
        }
        update();
        return error;
    }

    private float meanSquaredError(float[] outputs, float[] expectedOutputs) {
        return squaredError(outputs, expectedOutputs) / outputs.length;
    }

    private float squaredError(float[] outputs, float[] expectedOutputs) {
        if (outputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("The outputs and expected outputs arrays are not the same size!");
        }
        float error = 0;
        for (int i = 0; i < outputs.length; i++) {
            error += Math.pow(outputs[i] - expectedOutputs[i], 2);
        }
        return error;
    }

    private float[] squaredErrorPD(float[] outputs, float[] expectedOutputs) {
        if (outputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("The outputs and expected outputs arrays are not the same size!");
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

    private float sigmoidD(float x) {
        return (float)(Math.exp(x) / Math.pow(1 + Math.exp(x), 2));
    }

    private float elu(float x) {
        if (x >= 0) {
            return x;
        } else {
            return (float)(Math.exp(x) - 1);
        }
    }

    private float eluD(float x) {
        if (x >= 0) {
            return 1;
        } else {
            return (float)Math.exp(x);
        }
    }

    public void update() {
        for (int i = 1; i < layers.length; i++) {
            layers[i].update();
        }
    }

    public Layer layer(int index) {
        return layers[index];
    }

    public int length() {
        return layers.length;
    }

    public String toString() {
        String s = new String();
        for (int i = 0; i < layers.length; i++) {
            s += layers[i];
            if (i < layers.length - 1) {
                s += "\n";
            }
        }
        return s;
    }
}

class Layer {
    private final Neuron[] neurons;

    public Layer(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("Cannot create a layer smaller than 1 neuron!");
        }
        this.neurons = new Neuron[size];
        for (int i = 0; i < this.neurons.length; i++) {
            this.neurons[i] = new Neuron();
        }
    }

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

    private float[] he(int size) {
        float[] weights = new float[size];
        Random r = new Random();
        float stddev = (float)Math.sqrt(2.0 / size);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float)r.nextGaussian(0, stddev);
        }
        return weights;
    }

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

    public void update() {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].update();
        }
    }

    public Neuron neuron(int index) {
        return neurons[index];
    }

    public int length() {
        return neurons.length;
    }

    public String toString() {
        String s = new String();
        for (int i = 0; i < neurons.length; i++) {
            s += neurons[i];
            if (i < neurons.length - 1) {
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
    private float cacheBias;
    private final float[] weights;
    private final float[] cacheWeights;
    private float gradient;

    public Neuron() {
        this.weights = null;
        this.cacheWeights = null;
    }

    public Neuron(float[] weights, float bias) {
        this.bias = bias;
        this.weights = new float[weights.length];
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = weights[i];
        }
        this.cacheWeights = new float[this.weights.length];
    }

    public void update() {
        bias = cacheBias;
        for (int i = 0; i < weights.length; i++) {
            weights[i] = cacheWeights[i];
        }
        gradient = 0;
        cacheBias = 0;
        Arrays.fill(cacheWeights, 0);
    }

    public void setValue(float value) {
        this.value = value;
    }

    public void setActivation(float activation) {
        this.activation = activation;
    }

    public void setBias(float bias) {
        cacheBias = bias;
    }

    public void setWeight(int index, float weight) {
        cacheWeights[index] = weight;
    }

    public void setGradient(float gradient) {
        this.gradient = gradient;
    }

    public float getValue() {
        return value;
    }

    public float getActivation() {
        return activation;
    }

    public float getBias() {
        return bias;
    }

    public float getWeight(int index) {
        return weights[index];
    }

    public float getGradient() {
        return gradient;
    }

    public String toString() {
        return String.valueOf(activation);
    }
}