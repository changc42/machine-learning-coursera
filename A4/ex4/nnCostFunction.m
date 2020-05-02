function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
act1= [ones(m,1) X];%5000x401

act2 = [ones(1,m) ;sigmoid(Theta1*act1')];%26x5000
act3 = sigmoid(Theta2*act2); %10x5000

y_bin = [1:num_labels];
y_bin = repmat(y_bin,m,1);

for i = 1:m
    for j = 1:num_labels
        y_bin(i,j) = y_bin(i,j)==y(i);
    endfor
endfor
%y_bin m x num_labels

a= -y_bin;
b=log(act3);
c= 1-y_bin;
d= log(1-act3);

temp = a.*b'-c.*d'; %m x k
temp2 = zeros(m:1);
for i = 1:m
    temp2(i) = sum(temp(i,:));
endfor


J = (1/m)*sum(temp2);
reg_theta1 = Theta1(:,(2:size(Theta1,2)));
reg_theta1_sum=0;
for j = 1:size(reg_theta1,1);
    reg_theta1_sum += sum(reg_theta1(j,:).^2);
end;

reg_theta2 = Theta2(:,(2:size(Theta2,2)));
reg_theta2_sum=0;
for j = 1:size(reg_theta2,1);
    reg_theta1_sum += sum(reg_theta2(j,:).^2);
end;
J_reg = (lambda/(2*m))*(reg_theta1_sum+reg_theta2_sum);
J += J_reg;





% -------------------------------------------------------------
Delta1 = zeros(hidden_layer_size, (input_layer_size + 1));
Delta2 = zeros(num_labels, (hidden_layer_size + 1));
for t = 1:m
    act1 = [1 X(t,:)]; %1 x m+1
    act2 = [1 ;sigmoid(Theta1*act1')]; %hiddenlayer size +1 x 1
    act3 = sigmoid(Theta2*act2); % outputs x 1

    delta3 = act3 - y_bin(t,:)';

    Delta2 = Delta2 + delta3*act2';

    
    
    
    
    delta2 = (Theta2(:,(2:end)))'*delta3.*sigmoidGradient(Theta1*act1');


    Delta1 = Delta1 + delta2*act1;
end

D1 = (1/m).*Delta1;
D2 = (1/m).*Delta2;

Theta1_grad =D1+[zeros(hidden_layer_size,1) (lambda/m)*Theta1(:,(2:end))];
Theta2_grad =D2+[zeros(num_labels,1) (lambda/m)*Theta2(:,(2:end))];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end