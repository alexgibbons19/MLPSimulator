function [netw, iteration, RMSE,RMSE_val] = LearningMLPWithValidation(Input, ValidationSet, nhiddenneurons, noutputs, errorlimit, iterationslimit, OutputFlag, EarlyStop)
    
    % nhiddenneurons - # of hidden neurons in a single hidden layer
    % noutputs - # of network outputs (# of output neurons)
    % errorlimit - tolerance threshold for the error
    % iterationslimint - max # of iterations
    % OptputFlag = k, only every kth learning iteration will be displayed
    
    % netw - weights
    % iterations - final # of iterations
    % RMSE - final learning RMSE
    

    %Input is a matrix containing a test set;
    % Each row is a sample - inputs followed by a desired output
    
    A = Input;
     
    % N is now the number of learning samples
    % ninputs is now the number of inputs
    [N,ninputs]=size(A);
    ninputs=ninputs-1;

    % extract validation set
    [N_val,ninputs_val] = size(ValidationSet);
    ninputs_val=ninputs_val-1;
    inputs_val = ValidationSet(:,1:ninputs_val);
    targets_val = ValidationSet(:,ninputs_val + 1);
    
    % An array to store actual outputs after learning
    ActualOutputs = zeros(1, N);
    ActualOutputs_val = zeros(1, N_val);
    
%     ActualOutputs_val_netw = zeros(1, N_val);

    % extraction of input samples (only inputs)
    inputs=A(:,1:ninputs);
    % extraction of the desired outputs (assume that there is only 1 output
    % neuorn)
    targets=A(:,ninputs+1);
    
    

    % wsize is the length of an array for storing all weights of this network.
    % There are contains (ninputs+1)*nhiddenneurons) weights of hidden neuurons
    % and (nhiddenneurons*noutputs) weights of output neurons
    wsize = ((ninputs+1)*nhiddenneurons)+(nhiddenneurons*noutputs);% +1;
    %creates a neural network with ninputs inputs nhiddenneurons neurons in a
    %single hidden layer and noutputs neurons in the output layer
    net=CreateNN(ninputs,nhiddenneurons,noutputs);
    netw = net.w;
    
    % Set a counter of iterations
    iteration=0;

    % Validation RMSE
    RMSE_val = 10;
    best_val_RMSE=RMSE_val;
    iters_without_improvement=0;
    
    
    % A main loop with the learning process
    while (iteration<=iterationslimit)&&(RMSE_val > errorlimit)
    % learning continues as long as RMSE_Val>errorlimit and iteration<=iterationslimit  
        % increment intreations
        iteration=iteration+1;

        % Evaluation of RMSE for the entire learning set
        % a for loop over all learning samples
         for j=1:N
              % calculation of the actual output of the network for the j-th
              % sample
              output  = EvalNN( inputs(j,:),netw,ninputs,nhiddenneurons,noutputs );
              % Accumulation of actual outputs
              ActualOutputs(j) = output;
         end
        % MSE over all learning samples
         error = sum((ActualOutputs - targets').^2)/N;
  
        % RMSE
         RMSE = sqrt(error);
        
        % Validation RMSE
        for j=1:N_val
            output_val = EvalNN(inputs_val(j,:), netw, ninputs, nhiddenneurons, noutputs);
            ActualOutputs_val(j) = output_val;
        end
        
        error_val = sum((ActualOutputs_val - targets_val').^2) / N_val;
        RMSE_val = sqrt(error_val);

        if mod(iteration,OutputFlag)==0
           fprintf(' Iterations = %f \n',iteration); 
           fprintf(' Training error = %f \n',RMSE); 
           fprintf(' Validation error = %f \n',RMSE_val); 
        end
       
        % Early Stop
        if RMSE_val < best_val_RMSE
          best_val_RMSE=RMSE_val;
          iters_without_improvement=0;
        else
          iters_without_improvement=iters_without_improvement+1;
       end
       
       if(iters_without_improvement >= EarlyStop)
           fprintf('No validation error improvement in %d iterations, breaking to prevent overfitting\n',EarlyStop);
           break
       end

        % if RMSE dropped below errorlimit, then the learning process converged
        if RMSE_val <= errorlimit
            break  % and we get out of the while loop
        end

        % otherwise we start correction of the weights

        % a for loop over all learning samples
        for j=1:N
             % backpropagation and correction of the weights
             netw  = BackProp(netw,inputs(j,:),targets(j,:),ninputs,nhiddenneurons,noutputs); 
        end
    end

    %test newtork function
    % final results of the learning process
    fprintf(' Iterations = %7d \n',iteration); 
     for j=1:N
              output  = EvalNN( inputs(j,:),netw,ninputs,nhiddenneurons,noutputs );
              ActualOutputs(j) = output; 
     end
    
    for j=1:N_val
            output_val = EvalNN(inputs_val(j,:), netw, ninputs, nhiddenneurons, noutputs);
            ActualOutputs_val(j) = output_val;
    end

    final_val_err=sum((ActualOutputs_val - targets_val').^2) / N_val;
    final_val_RMSE=sqrt(final_val_err);

    disp(['Validation Error=',num2str(error_val)]);
    disp(['Final Validation Error=',num2str(final_val_err)]);
    disp(['Final Validation RMSE=',num2str(final_val_RMSE)]);

    
%      display(['Error= ',num2str(error)]);
%      figure (1);
%      title('RMSE')
%      hold off
%      plot(targets,'or'); 
%      hold on
%      plot(ActualOutputs, '*g');
%      
%      display(['Validation Error= ',num2str(error_val)]);
%      figure (2);
%      title('Val_Error')
%      hold off
%      plot(targets_val,'or'); 
%      hold on
%      plot(ActualOutputs_val, '*g');
end