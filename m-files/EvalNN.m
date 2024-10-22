function [ output ] = EvalNN( inputs,individual,ninputs,nhiddenneurons,noutputs )

% A Matlab implementation of MLF with
% backpropagation training with momentum.
% Demo
% 

ninputs = ninputs+1; % +1 for bias node

N = numel(individual);

wi = individual(1:ninputs*nhiddenneurons);
wi = reshape(wi,ninputs,nhiddenneurons);

wo = individual(ninputs*nhiddenneurons+1:N);
wo = reshape(wo,nhiddenneurons,noutputs);

%input activations
ai     = [inputs 1]; % 1 for bias node;

% hidden activations
sum    = ai * wi;
ah     = tanh(sum);

% output activations
sum    =  ah * wo;
output = tanh(sum);
