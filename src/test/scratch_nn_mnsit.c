#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>


#define IMAGE_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049
#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define IMAGE_SIZE 784 // 28x28 pixels

#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.1
#define EPOCHS 100


// Sigmoid activation function
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// Softmax activation function
void softmax(double *x, int n) {
    double max = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    double sum = 0;
    for (int i = 0; i < n; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// Forward propagation
void forward(double *input, double input_hidden_weights[][HIDDEN_SIZE], 
             double *hidden_output_weights, double *hidden, double *output) {
    // Calculate hidden layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = 0;
        for (int j = 0; j < IMAGE_SIZE; j++) {
            sum += input[j] * input_hidden_weights[j][i];
        }
        hidden[i] = sigmoid(sum);
    }

    // Calculate output layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += hidden[j] * hidden_output_weights[j * OUTPUT_SIZE + i];
        }
        output[i] = sum; // Before applying softmax
    }

    // Apply softmax to output layer
    softmax(output, OUTPUT_SIZE);
}

// Backpropagation
void backward(double *input, double *target, double *output, 
              double input_hidden_weights[][HIDDEN_SIZE], 
              double *hidden_output_weights, double *hidden) {
    // Calculate output layer error
    double output_error[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = target[i] - output[i];
    }

    // Calculate hidden layer error
    double hidden_error[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double error = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            error += output_error[j] * hidden_output_weights[i * OUTPUT_SIZE + j];
        }
        hidden_error[i] = error * sigmoid_derivative(hidden[i]);
    }

    // Update hidden to output weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_output_weights[i * OUTPUT_SIZE + j] += LEARNING_RATE * output_error[j] * hidden[i];
        }
    }

    // Update input to hidden weights
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            input_hidden_weights[i][j] += LEARNING_RATE * hidden_error[j] * input[i];
        }
    }
}

// Function to read IDX3 format image file
void read_idx3_file(const char *filename, uint8_t ***images, int *num_images, int *image_size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read magic number
    int magic_number;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = __builtin_bswap32(magic_number); // Convert to big endian if necessary
    if (magic_number != IMAGE_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid magic number\n");
        exit(EXIT_FAILURE);
    }

    // Read number of images
    fread(num_images, sizeof(*num_images), 1, file);
    *num_images = __builtin_bswap32(*num_images); // Convert to big endian if necessary

    // Read image dimensions
    int rows, cols;
    fread(&rows, sizeof(rows), 1, file);
    fread(&cols, sizeof(cols), 1, file);
    rows = __builtin_bswap32(rows); // Convert to big endian if necessary
    cols = __builtin_bswap32(cols); // Convert to big endian if necessary
    *image_size = rows * cols;

    // Allocate memory for images
    *images = malloc(*num_images * sizeof(uint8_t *));
    for (int i = 0; i < *num_images; i++) {
        (*images)[i] = malloc(*image_size * sizeof(uint8_t));
        fread((*images)[i], sizeof(uint8_t), *image_size, file);
    }

    fclose(file);
}

// Function to read IDX1 format label file
void read_idx1_file(const char *filename, uint8_t **labels, int *num_labels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read magic number
    int magic_number;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = __builtin_bswap32(magic_number); // Convert to big endian if necessary
    if (magic_number != LABEL_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid magic number\n");
        exit(EXIT_FAILURE);
    }

    // Read number of labels
    fread(num_labels, sizeof(*num_labels), 1, file);
    *num_labels = __builtin_bswap32(*num_labels); // Convert to big endian if necessary

    // Allocate memory for labels
    *labels = malloc(*num_labels * sizeof(uint8_t));
    fread(*labels, sizeof(uint8_t), *num_labels, file);

    fclose(file);
}

// Function to normalize pixel values of images
void normalize_images(uint8_t **images, int num_images, int image_size) {
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < image_size; j++) {
            images[i][j] /= 255.0; // Normalize to range [0, 1]
        }
    }
}

// Function to flatten 2D images to 1D vectors
void flatten_images(uint8_t **images, int num_images, int image_size, double ***flattened_images) {
    *flattened_images = malloc(num_images * sizeof(double *));
    for (int i = 0; i < num_images; i++) {
        (*flattened_images)[i] = malloc(image_size * sizeof(double));
        for (int j = 0; j < image_size; j++) {
            (*flattened_images)[i][j] = (double)images[i][j]; // Cast to double
        }
    }
}



int main() {

    uint8_t **training_images, **test_images;
    uint8_t *training_labels, *test_labels;
    int num_training_images, num_test_images, image_size;

    // Read training images and labels
    read_idx3_file("train-images.idx3-ubyte", &training_images, &num_training_images, &image_size);
    read_idx1_file("train-labels.idx1-ubyte", &training_labels, &num_training_images);

    // Read test images and labels
    read_idx3_file("t10k-images.idx3-ubyte", &test_images, &num_test_images, &image_size);
    read_idx1_file("t10k-labels.idx1-ubyte", &test_labels, &num_test_images);

    // Preprocess training and test images
    normalize_images(training_images, num_training_images, image_size);
    normalize_images(test_images, num_test_images, image_size);

    double **flattened_training_images, **flattened_test_images;
    flatten_images(training_images, num_training_images, image_size, &flattened_training_images);
    flatten_images(test_images, num_test_images, image_size, &flattened_test_images);

    // Define weights
    double input_hidden_weights[IMAGE_SIZE][HIDDEN_SIZE];
    double hidden_output_weights[HIDDEN_SIZE * OUTPUT_SIZE];

    // Initialize weights randomly
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            input_hidden_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        hidden_output_weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }



	// Start timing
    clock_t start = clock();
    // Training loop
    int num_epochs = EPOCHS;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;

    // Iterate over training data
        for (int i = 0; i < num_training_images; i++) {
        // Forward propagation
            double hidden[HIDDEN_SIZE];
            double output[OUTPUT_SIZE];
            forward(flattened_training_images[i], input_hidden_weights, hidden_output_weights, hidden, output);

        // Calculate loss (cross-entropy)
        double target[OUTPUT_SIZE] = {0}; // Initialize target array with zeros
        target[training_labels[i]] = 1;   // Set the target for the true class to 1
        double loss = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            loss -= target[j] * log(output[j]); // Cross-entropy loss
        }
        total_loss += loss;

        // Backpropagation
        backward(flattened_training_images[i], target, output, input_hidden_weights, hidden_output_weights, hidden);

        // Update weights
        // Weights are updated during backpropagation
    }

    // Print average loss for the epoch
    printf("Epoch %d, Loss: %f\n", epoch+1, total_loss / num_training_images);
}

	// End timing
    clock_t end = clock();
    double elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Training time: %lf seconds\n", elapsed_time);

    // Evaluate the trained model on the test dataset
    int correct_predictions = 0;
    
    for (int i = 0; i < num_test_images; i++) {
    // Forward propagation
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    forward(flattened_test_images[i], input_hidden_weights, hidden_output_weights, hidden, output);

    // Find the index of the output with the highest probability
    int predicted_label = 0;
    double max_prob = output[0];
    for (int j = 1; j < OUTPUT_SIZE; j++) {
        if (output[j] > max_prob) {
            max_prob = output[j];
            predicted_label = j;
        }
    }

    // Check if the predicted label matches the true label
    if (predicted_label == test_labels[i]) {
        correct_predictions++;
    }
}

// Calculate accuracy
double accuracy = (double)correct_predictions / num_test_images * 100.0;
printf("Accuracy on test dataset: %.2f%%\n", accuracy);

    // Free memory
    for (int i = 0; i < num_training_images; i++) {
        free(training_images[i]);
    }
    free(training_images);
    free(training_labels);

    for (int i = 0; i < num_test_images; i++) {
        free(test_images[i]);
    }
    free(test_images);
    free(test_labels);

    // Free memory for flattened images
    for (int i = 0; i < num_training_images; i++) {
        free(flattened_training_images[i]);
    }
    free(flattened_training_images);

    for (int i = 0; i < num_test_images; i++) {
        free(flattened_test_images[i]);
    }
    free(flattened_test_images);

    return 0;
}