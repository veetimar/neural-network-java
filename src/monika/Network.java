package monika;
import java.util.*;
//TODO: backward (sgd)
public class Network {
    private final Layer[] layers;
    private final float learningRate = 0.01f;

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
    }

    public float[] forward(float[] inputs) {
        if (inputs.length != layers[0].getLength()) {
            throw new IllegalArgumentException("The size of the inputs array does not match the size of the input layer!");
        }
        for (int i = 0; i < inputs.length; i++) {
            layers[0].getNeuron(i).setValue(inputs[i]);
        }
        float[] outputs = new float[layers[layers.length - 1].getLength()];
        for (int i = 1; i < layers.length; i++) {
            boolean output = i == layers.length - 1;
            for (int j = 0; j < layers[i].getLength(); j++) {
                float value = 0;
                for (int k = 0; k < layers[i - 1].getLength(); k++) {
                    value += layers[i - 1].getNeuron(k).getValue() * layers[i].getNeuron(j).getWeight(k);
                }
                value += layers[i].getNeuron(j).getBias();
                if (!output) {
                    value = elu(value);
                    layers[i].getNeuron(j).setValue(value);
                } else {
                    value = sigmoid(value);
                    layers[i].getNeuron(j).setValue(value);
                    outputs[j] = value;
                } 
            }
        }
        return outputs;
    }
 
    public float[][] train(float[][] inputs, float[][] outputs, int times) {
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("Different number of input and output examples!");
        }
        float[][] errors = new float[times][inputs.length];
        for (int i = 0; i < times; i++) {
            for (int j = 0; j < inputs.length; j++) {
                errors[i][j] = backward(inputs[j], outputs[j]);
            }
        }
        return errors;
    }

    public float backward(float[] inputs, float[] expectedOutputs) {
        if (inputs.length != layers[0].getLength()) {
            throw new IllegalArgumentException("The size of the inputs array does not match the size of the input layer!");
        } else if (expectedOutputs.length != layers[layers.length - 1].getLength()) {
            throw new IllegalArgumentException("The size of the expected outputs array does not match the size of the output layer!");
        }
        float[] outputs = forward(inputs);
        float error = meanSquaredError(outputs, expectedOutputs);


        for (int i = layers.length - 1;i > 0; i-- ) {

        }


        update();
        return error;
    }

    private float meanSquaredError(float[] outputs, float[] expectedOutputs) {
        if (outputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("The Outputs and expected outputs arrays are not the same size!");
        }
        float x = 0;
        for (int i = 0; i < outputs.length; i++) {
            x += Math.pow(outputs[i] - expectedOutputs[i], 2);
        }
        return 1.0f / outputs.length * x;
    }

    private void update() {
        for (int i = 1; i < layers.length; i++) {
            layers[i].update();
        }
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

    public Layer getLayer(int index) {
        return layers[index];
    }

    public int getLength() {
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
            throw new IllegalArgumentException("Illegal parameter for the size of the previous layer!");
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

    private float[] he(int inputs) {
        float[] weights = new float[inputs];
        Random r = new Random();
        float stddev = (float)Math.sqrt(2.0 / inputs);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float)r.nextGaussian(0, stddev);
        }
        return weights;
    }

    private float[] xavier(int inputs) {
        float[] weights = new float[inputs];
        Random r = new Random();
        float min = (float)(-1.0 / Math.sqrt(inputs));
        float max = (float)(1.0 / Math.sqrt(inputs));
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

    public Neuron getNeuron(int index) {
        return neurons[index];
    }

    public int getLength() {
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
    private float bias;
    private float cacheBias;
    private float[] weights;
    private float[] cacheWeights;
    private float gradient;

    public Neuron() {
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
        return String.valueOf(value);
    }
}