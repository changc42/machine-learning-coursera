

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('testData.txt');
X = data(:, [1, 2]); y = data(:, 3);
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
[cost,grad] = costFunction(initial_theta, X, y);