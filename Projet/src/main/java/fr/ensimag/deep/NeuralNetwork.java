package fr.ensimag.deep;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import fr.ensimag.deep.layers.AbstractLayer;
import fr.ensimag.deep.layers.StandardLayer;
import fr.ensimag.deep.serialization.json.InitializerFactory;
import fr.ensimag.deep.serialization.json.LayerFactory;
import fr.ensimag.deep.serialization.json.NetworkDescription;
import fr.ensimag.deep.serialization.json.SerializedLayer;
import fr.ensimag.deep.trainer.initializer.INewtorkInitializer;
import lombok.Getter;
import lombok.Setter;


public class NeuralNetwork {
    /** Ordered list of layers of this neural network */
    @Getter @Setter private List<AbstractLayer> layers;
    @Getter private int inputSize;
    @Getter private int batchSize;
     
    @Getter @Setter private INewtorkInitializer initializer = null;

    /** 
     * Creates an empty neural networks
     */
    public NeuralNetwork(int batchSize)
    {
        this.layers = new ArrayList<AbstractLayer>();
        this.batchSize = batchSize;
    }

    /**
     * Returns the output size of the network (the dimension of the output space 
     * of the function implemented by the network)
     * @return the output size
     */
    public int getOutputSize()
    {
        throw new UnsupportedOperationException("not yet implemented");
    }

    /**
     * Adds a new layer at the end of all the already added layers
     * @param l
     * @return the current Neural Network. It allows to easily chain the addLayer calls
     */
    public NeuralNetwork addLayer(AbstractLayer l)
    {
        this.layers.add(l);
        return this;
    }

    /** 
     * Randomly initialize the neural network using the associated initializer (private attribute)
     */
    public void initialize()
    {
        if (this.initializer == null)
            throw new RuntimeException("No intitializer has been defined");
        else
            initializer.init(this);
    }

    /** returns the output of the neural network. By definition, the output is
     * the activation of the last layer
     * @return the output
     */
    public SimpleMatrix getOutput()
    {
        throw new UnsupportedOperationException("not yet implemented");
    }

    /**
     * Calculate the output(s) of the neural network corresponding to the associated input(s)
     * The input matrix is of size mxn where m is the input size of the network (the input space
     * dimension of the corresponding function) and n is the number of input vectors
     * @param input the input vectors
     * @return the output vectors. It is a m'xn matrix, where m' is the output dimension of the 
     *         correspondig function, and n the number of input vectors
     */
    public SimpleMatrix propagate(SimpleMatrix input)
    {   
        SimpleMatrix result= new SimpleMatrix(layers.get(layers.size()).getBiasSize(), input.getNumCols());
        AbstractLayer currentLayer = layers.get(0);
        currentLayer.propagate(input);
        
        for(int i=1; i<layers.size(); ++i){
            SimpleMatrix output =layers.get(i).getOutput();
            currentLayer = layers.get(i);
            currentLayer.propagate(output);
            result.setColumn(i, output);
        }

        return result;  
        
      
    }

    /**
     * Specify the batch size (the number of input vectors in a row to compute the output or in a learning step)
     * This batch size must be propagated to every layers
     * @param batchSize the batch size
     */
    public void setBatchSize(int batchSize)
    {
        this.batchSize=batchSize;   
    }

    /**
     * Backpropagate through the layers (from the last to the first) the gradient computation 
     * and update the weights accordingly
     * @param lossFunctionGradient a matrix corresponding to the derivative of the loss between 
     *                             the expected output and the computed output value
     */
    public void learn(SimpleMatrix lossFunctionGradient)
    {
        this.backpropagateAndUpdate(this.getOutputLayer(), lossFunctionGradient);
        for (int i=getNbLayers()-2; i>=0; i--)
            this.backpropagateAndUpdate(layers.get(i), layers.get(i+1).getWeightedError());
    }

    /**
     * Backpropagate the error to compute the weights and bias gradients on a layer and updates its parameters based on the gradients
     * @param layer the layer
     * @param layerError the output error of this layer
     */
    private void backpropagateAndUpdate(AbstractLayer layer, SimpleMatrix layerError) {
        layer.backpropagate(layerError);
        layer.updateParameters();
    }

    /**
     * Returns the output layer of the Neural Netork (the last layer that has been inserted)
     * @return the Output Layer
     */
    private AbstractLayer getOutputLayer()
    {
        return this.layers.get(this.layers.size()-1);
    }

    /**
     * Number of layers in the current neural network
     * @return the number of layers
     */
    public int getNbLayers()
    {
        return this.layers.size();
    }

    /**
     * 
     * Saves the current neural network as a JSON file. The JSON format is specified 
     * by the NetworkDescription java object (a Java class corresponding to the json
     * format chosen to represent a neural network)
     * @param filename the file name of the saved network
     * @return true is the network has been saved, false otherwise (no exception is thrown)
     */
   public boolean save(String filename) 
   {
        NetworkDescription dto = new NetworkDescription();

        // to be completed : initialize the dto object with the neural network caracteristics 

        // serialize the NetworkDescription object (dto) to its json representation
        try {
            ObjectMapper mapper = new ObjectMapper();
            String json = mapper.writerWithDefaultPrettyPrinter().
                                writeValueAsString(dto);
            BufferedWriter w = new BufferedWriter(new FileWriter(filename));

            w.write(json);
            w.close();
        } catch (JsonProcessingException e) {
            return false;
        } catch (IOException e) {
            return false;
        }

        return true;
    }

    /**
     * Creates a Neural Network from its json description in a file
     * @param filename the filename containing the json description
     * @return the Neural Network
     * @throws IOException
     */
    static public NeuralNetwork fromDescription(String filename) throws IOException
    {
        // Created the NetworkDescription object corresponding to the json representation in the file
        NeuralNetwork net = null;
        byte[] jsonData = Files.readAllBytes(Paths.get(filename));
		
		//create ObjectMapper instance
		ObjectMapper objectMapper = new ObjectMapper().configure(JsonParser.Feature.ALLOW_COMMENTS, true);;
		
		//convert json string to object
		NetworkDescription neuralDTO = objectMapper.readValue(jsonData, NetworkDescription.class);        

        // to do : Initialize the NeuralNetwork net object with the elements provided by the neuralDTO 
        // loaded from the json file

        return net;
    }
}
