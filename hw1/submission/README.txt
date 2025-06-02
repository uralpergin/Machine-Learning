First of all I define the Module.
In order to have multiple hidden layer, there is a for loop to apply activation functions to all hidden layers.

Then, I define the hyperparameter pool and variables in order to keep track of CI and accuracy values.

There is a nested for loop for all the parameters in order to implement the grid search.

Each configuration is runned 10 times and corresponding accuracy values are calculated for each run.

After a configuration iteration passes the confidence interval value is calculated and compared with the latest CI value.

There are 243 configurations and after the maximum CI value is calculated after 243 config iterations.

Best hyperparameters are determined according to the max CI value. (the parameters that gives the max CI value)

After that, train and validation data is concatenated and fine-tuned in this new dataset.

Lastly, it is tested on the test data set and last accuracy score and CI score is calculated according to the best parameters.
