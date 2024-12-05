package fr.ensimag.deep.trainer.initializer;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.ejml.simple.SimpleMatrix;

import fr.ensimag.deep.NeuralNetwork;
import fr.ensimag.deep.layers.AbstractLayer;

/**
 * Random initialization of weights
 */
public class XavierInitializer implements INewtorkInitializer {
    private NormalDistribution distribution = null;

    public XavierInitializer()
    {
        this.distribution = new NormalDistribution();
    }

   /** 
     * Random initialization of weights and bias of the layers of the network
     * Weights are drawn from a Gaussian distribution with standard deviation 
     * of sqrt(6/(n+m)) where n is the size of the previous layer and m the size
     * of the current layer
     * distribution.sample() draws a number from a gaussian distribution centered on 0
     * with a standard deviation of 1
     * @param neuralNetwork
     */
    public void init(NeuralNetwork neuralNetwork)
     {
        throw new UnsupportedOperationException("not yet implemented");
     }
}
