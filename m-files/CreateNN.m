function [ net ] = CreateNN( ninputs,nhiddenneurons,noutputs )
% A Matlab implementation of a MultiLayer perceptron with
% backpropagation training with momentum.
% 
% 
% by: José Antonio Martín H. <jamartinh [*AT*] fdi ucm es>
% 
% 
% Placed in the public domain. August 23, 2008

wsize   = ((ninputs+1)*nhiddenneurons)+(nhiddenneurons*noutputs); % +1 for bias;

% randomization of the random numbers generator based on the current
% time
rng('shuffle');
net.w   = rand(1,wsize) - 0.5;
net.ni  = ninputs;
net.nh  = nhiddenneurons;
net.no  = noutputs;