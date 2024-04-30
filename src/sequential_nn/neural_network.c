#include "neural_network.h"
#include <math.h>
#include <stdlib.h>

void forward(double input[], double weights_input_hidden[][INPUT_SIZE], 
             double hidden[], double weights_hidden_output[][HIDDEN_SIZE],
             double output[]) {
    // Calculate hidden layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input[j] * weights_input_hidden[i][j];
        }
        hidden[i] = sigmoid(sum);
    }

    // Calculate output layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += hidden[j] * weights_hidden_output[j][i];
        }
        output[i] = sigmoid(sum);
    }
}

void backward(double input[], double target[], double output[], 
              double hidden[], double weights_hidden_output[][HIDDEN_SIZE], 
              double weights_input_hidden[][INPUT_SIZE]) {
    // Calculate output layer error
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = target[i] - output[i];
    }

    // Calculate hidden layer error
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double error = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            error += output_error[j] * weights_hidden_output[i][j];
        }
        hidden_error[i] = error * sigmoid_derivative(hidden[i]);
    }

    // Update weights between hidden and output layers
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_hidden_output[j][i] += LEARNING_RATE * output_error[i] * hidden[j];
        }
    }

    // Update weights between input and hidden layers
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            weights_input_hidden[i][j] += LEARNING_RATE * hidden_error[i] * input[j];
        }
    }
}
