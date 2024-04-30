#include <stdio.h>
#include "neural_network.h"
#include <time.h>
#include <stdlib.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.1
#define EPOCHS 1000

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
