package fr.ensimag.deep.layers;

import org.ejml.simple.SimpleMatrix;

import fr.ensimag.deep.activators.IActivator;

public class StandardLayer extends AbstractLayer{
    private SimpleMatrix weights;
    private SimpleMatrix bias;
    //sum weight+bias
    private SimpleMatrix output;




    public StandardLayer(SimpleMatrix initialWeights, SimpleMatrix initialBias, IActivator activator)
    {
        super(initialBias.getNumRows(), initialWeights.getNumRows(), activator);
        this.weights=initialWeights;
        this.bias=initialBias;
        int layerSize=this.bias.getNumRows();
        this.output=new SimpleMatrix(layerSize,1);
    }


    
    
    @Override
    public void propagate(SimpleMatrix input) {
        int n = input.getNumRows();
        for (int k = 0; k < layerSize; ++k) {
            double sum = bias.get(k); 
            for (int j = 0; j < n; ++j) {
                sum += this.weights.get(j, k) * input.get(j, 0);
            }
            this.output.set(k, sum); 
        }
       

    }


    @Override
    public SimpleMatrix getActivation() {
        throw new UnsupportedOperationException("not yet implemented");
    }


    @Override
    public SimpleMatrix getOutput(){
        return this.output;
    }

    @Override
    public SimpleMatrix getWeightedError() {
        throw new UnsupportedOperationException("not yet implemented");
    }    

    @Override
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        // to be completed
    }

    @Override
    public void updateParameters() {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public void backpropagate(SimpleMatrix upperWeightedError) {
        throw new UnsupportedOperationException("not yet implemented");
    }
                

    @Override
    public SimpleMatrix getWeights() {
        return this.weights;
       
    }

    @Override
    public SimpleMatrix getBias() {
        return this.bias;
    }

    @Override
    public int getBiasSize(){
        return this.bias.getNumRows();
    }

    @Override
    public void setWeights(SimpleMatrix weights) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public void setBias(SimpleMatrix bias) {
        throw new UnsupportedOperationException("not yet implemented");
    }
}
