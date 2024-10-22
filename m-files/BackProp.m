function [ Weights ] = BackProp(Weights,inputs,targets,ninputs,nhiddenneurons,noutputs)
% A Matlab implementation of MLP-MLF with backpropagation learning with
% Weights - array with all weights (1D)
% inputs - array with inputs
% targets - array with desired outputs
% ninputs - # of inputs
% nhoddenneurons - # of hidden neurons in a single hidden layer
% noutputs - # of output neurons
% 
% Demo
% 


ninputs = ninputs+1; % +1 for bias node

N  = numel(Weights);

% wi - hidden neurons' weights
wi = Weights(1:ninputs*nhiddenneurons);
wi = reshape(wi,ninputs,nhiddenneurons);

% wo - output neuron(s) weights
wo = Weights(ninputs*nhiddenneurons+1:N);
wo = reshape(wo,nhiddenneurons,noutputs);

%input activations
ai     = [inputs 1]; % 1 for bias node;

% weighted sums for all hidden neurons
netW   = ai * wi;
% outputs of all hidden neurons
ah     = tanh(netW);

% weighted sums for all output neurons
netW   =  ah * wo;
% output of all output neurons = network output
ao     = tanh(netW);

% calculate errors for the output neurons
% (1.0-ao.^2) is a derivative of the tanh activation function
output_errors = (1.0-ao.^2) .* (targets - ao);

% backpropagation of the output neurons' errors to hidden layer neurons
error         = output_errors * wo';
% (1.0-ah.^2) is a derivative of the tanh activation function
hidden_errors = (1.0-ah.^2) .* error;

% update hidden neurons' weights
change  = reshape(ai,ninputs,1) * hidden_errors; 
wi = wi + (1/ninputs) .* change;

% update hidden neurons' outputs
% weighted sums for all hidden neurons
netW   = ai * wi;
% outputs of all hidden neurons
ah     = tanh(netW);

% update output neurons' weights
change  = reshape(ah,nhiddenneurons,1) * output_errors;
wo = wo + (1/(nhiddenneurons+1)) .* change;


Weights = [wi(:)' wo(:)'];
