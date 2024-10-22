function [netw, iteration, RMSE_train,RMSE_val] = IncLearningwValidation(Input, nhiddenneurons, noutputs, errorlimit, iterationslimit, ValidationInterval,ValidationData,EarlyStop)
    
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

    % An array to store actual outputs after learning
    ActualOutputs = zeros(1, N);

    % extraction of input samples (only inputs)
    inputs=A(:,1:ninputs);
    % extraction of the desired outputs (assume that there is only 1 output
    % neuorn)
    targets=A(:,ninputs+1);


    %inputs
    %targets

    % wsize is the length of an array for storing all weights of this network.
    % There are contains (ninputs+1)*nhiddenneurons) weights of hidden neuurons
    % and (nhiddenneurons*noutputs) weights of output neurons
    wsize = ((ninputs+1)*nhiddenneurons)+(nhiddenneurons*noutputs);% +1;
    %creates a neural network with ninputs inputs nhiddenneurons neurons in a
    %single hidden layer and noutputs neurons in the output layer
    net=CreateNN(ninputs,nhiddenneurons,noutputs);
    netw = net.w;

    %%%% EXTRACTING THE VALIDATION SET


    valInputs = ValidationData(:,1:ninputs);
    valTargets = ValidationData(:,ninputs+1);
    [N_val,M_val]=size(ValidationData);

    

    % Set initial error to start the while loop with learning
    RMSE_train=10;
    RMSE_val=10;
    best_val_RMSE=RMSE_val;
    

    % Set a counter of iterations
    iteration=0;
    iters_without_improvement=0;

    valOutputs=zeros(1,size(ValidationData,1));

    % A main loop with the learning process
    while (iteration<=iterationslimit)&&(RMSE_val>errorlimit)
    % learning continues as long as error>errorlimit and iteration<=iterationslimit  
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

             netw = BackProp(netw,inputs(j,:),targets(j,:),ninputs,nhiddenneurons,noutputs);
        end

        % MSE over all learning samples
        error = sum((ActualOutputs - targets').^2)/N;
        % RMSE
        RMSE_train = sqrt(error);

        for k=1:size(ValidationData,1)
            valOutputs(k)=EvalNN(valInputs(k,:),netw,ninputs,nhiddenneurons,noutputs);
        end

       valError = sum((valOutputs - valTargets').^2) / N_val;
       RMSE_val=sqrt(valError);


       if mod(iteration,ValidationInterval)==0
           fprintf(' Iterations = %f \n',iteration); 
           fprintf(' Training error = %f \n',RMSE_train); 
           fprintf(' Validation error = %f \n',RMSE_val); 
       end
       
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
       
        if (RMSE_val <= errorlimit)
            break  % and we get out of the while loop
        end
    end
    
    
    %test newtork function
    % final results of the learning process
    fprintf(' Iterations = %7d \n',iteration); 
    for j=1:N
             output  = EvalNN( inputs(j,:),netw,ninputs,nhiddenneurons,noutputs );
             ActualOutputs(j) = output;
             %disp([' Inputs [',num2str(inputs(j,:)),'] --> outputs:  [',num2str(output),']']) 

    end
%     display(['Error= ',num2str(error)]);
%     figure (1);
%     hold off
%     plot(targets,'or'); 
%     hold on
%     plot(ActualOutputs, '*g');
end