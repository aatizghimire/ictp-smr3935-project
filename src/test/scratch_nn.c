#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.1
#define EPOCHS 10000

// Sigmoid activation function
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// Forward propagation
void forward(double input[INPUT_SIZE], double weights_input_hidden[HIDDEN_SIZE][INPUT_SIZE], 
             double hidden[HIDDEN_SIZE], double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE],
             double output[OUTPUT_SIZE]) {
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

// Backpropagation
void backward(double input[INPUT_SIZE], double target[OUTPUT_SIZE], double output[OUTPUT_SIZE], 
              double hidden[HIDDEN_SIZE], double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE], 
              double weights_input_hidden[HIDDEN_SIZE][INPUT_SIZE]) {
    double output_error[OUTPUT_SIZE];
    double hidden_error[HIDDEN_SIZE];

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

int main() {
    // Initialize weights randomly
    double weights_input_hidden[HIDDEN_SIZE][INPUT_SIZE];
    double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            printf("weights_input_hidden[%d][%d] = %lf\n", i, j, weights_input_hidden[i][j]);
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_hidden_output[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            printf("weights_hidden_output[%d][%d] = %lf\n", i, j, weights_hidden_output[i][j]);
        }
    }

    // Sample training data (XOR function)
    double training_data[4][INPUT_SIZE] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double training_labels[4][OUTPUT_SIZE] = {{0}, {1}, {1}, {0}};

	// Start timing
    clock_t start = clock();
	
    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double error = 0;
        for (int i = 0; i < 4; i++) {
            double input[INPUT_SIZE];
            double target[OUTPUT_SIZE];
            double hidden[HIDDEN_SIZE];
            double output[OUTPUT_SIZE];

            // Copy input and target
            for (int j = 0; j < INPUT_SIZE; j++) {
                input[j] = training_data[i][j];
            }
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                target[j] = training_labels[i][j];
            }

            // Forward propagation
            forward(input, weights_input_hidden, hidden, weights_hidden_output, output);

            // Backpropagation
            backward(input, target, output, hidden, weights_hidden_output, weights_input_hidden);

            // Calculate error
            error += 0.5 * ((target[0] - output[0]) * (target[0] - output[0]));
        }

        // Print error every 100 epochs
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch %d, Error: %lf\n", epoch + 1, error);
        }
    }
	
	// End timing
    clock_t end = clock();
    double elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Training time: %lf seconds\n", elapsed_time);

    // Test the trained network
    printf("Testing the trained network...\n");
    for (int i = 0; i < 4; i++) {
        double input[INPUT_SIZE];
        double hidden[HIDDEN_SIZE];
        double output[OUTPUT_SIZE];

        // Copy input
        for (int j = 0; j < INPUT_SIZE; j++) {
            input[j] = training_data[i][j];
        }

        // Forward propagation
        forward(input, weights_input_hidden, hidden, weights_hidden_output, output);

        // Print input, output, and target
        printf("Input: %lf, %lf, Output: %lf, Target: %lf\n", input[0], input[1], output[0], training_labels[i][0]);
    }

    return 0;
}