function [RMSE,ClassifRate] = TestingMLP2(Input, nhiddenneurons, netw)
% This program performs test of a trained MLP-MLF with 1 hidden layer
% and a binary output
% It takes inputs with desired outputs, calculates actual outputs and RMSE
% between actual and desired outputs

% Input contains a testing set (last column is a desired output)
% netw is an array of weights
% nhiddenneurons is the # of neurons in a single hidden layer

noutputs = 1; % # of output neurons

[N,ninputs]=size(Input);

% Desired outputs
targets=Input(:,ninputs);

ninputs=ninputs-1;

% inputs contain only inputs
inputs = Input(: , 1:ninputs);

% An array to store actual outputs after learning
ActualOutputs = zeros(1, N);

% N is now the number of learning samples
% ninputs is now the number of inputs

% a for loop over all learning samples
for j=1:N
         % calculation of the actual output of the network for the j-th
         % sample
         output  = EvalNN( inputs(j,:),netw,ninputs,nhiddenneurons,noutputs );
         
             % interpretation of output in the case of a binary output
             % any non-negative output --> 0.5
             % any negative output --> - 0.5
             if (output>=0) 
                 output = 0.5;
             else
                 output = -0.5;
             end
         
         
         % Accumulation of actual outputs
         ActualOutputs(j) = output;
%         disp([' Inputs [',num2str(inputs(j,:)), ' ] --> Actual Output:  [',num2str(output),']', '] --> Desired Output:  [',num2str(targets(j)),']'] )
         
end
% MSE over all testing samples
error = sum((ActualOutputs - targets').^2)/N;
% RMSE
RMSE = sqrt(error);

% Evaluation of the classification rate
% Results will contain 1 if there is no error and 0 otherwise
Results = (ActualOutputs' == targets);
NumOfCorrectOutputs = sum (Results);
ClassifRate = (NumOfCorrectOutputs / N) * 100;

fprintf('Prediction/Recognition Error = %f \n', RMSE);
fprintf('Classification Rate = %f \n', ClassifRate);
% 
% figure ();
% hold off
% plot(targets,'Or'); 
% hold on
% plot(ActualOutputs, '*b');

end

