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
#define EPOCHS 10

// Sigmoid activation function
__device__ double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivative of the sigmoid function
__device__ double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// Softmax activation function
__device__ void softmax(double *x, int n) {
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

// Forward propagation CUDA kernel
__global__ void forwardPropagationKernel(double *input_images, double *input_hidden_weights,
                                         double *hidden_output_weights, double *hidden_activations,
                                         double *output_activations, int num_images, int image_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    if (idx < num_images) {
        // Calculate hidden layer activations
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double sum = 0.0;
            for (int j = 0; j < IMAGE_SIZE; j++) {
                sum += input_images[idx * image_size + j] * input_hidden_weights[j * HIDDEN_SIZE + i];
            }
            hidden_activations[idx * HIDDEN_SIZE + i] = sigmoid(sum);
        }

        // Calculate output layer activations
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            double sum = 0.0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += hidden_activations[idx * HIDDEN_SIZE + j] * hidden_output_weights[j * OUTPUT_SIZE + i];
            }
            output_activations[idx * OUTPUT_SIZE + i] = sum; // Before applying softmax
        }

        // Apply softmax to output layer activations
        softmax(output_activations + idx * OUTPUT_SIZE, OUTPUT_SIZE);
    }
}

// Backward propagation CUDA kernel
__global__ void backwardPropagationKernel(double *input_images, double *target_labels,
                                          double *hidden_activations, double *output_activations,
                                          double *input_hidden_weights, double *hidden_output_weights,
                                          int num_images, int image_size, double learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    if (idx < num_images) {
        // Calculate output layer error
        double output_error[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_error[i] = target_labels[idx * OUTPUT_SIZE + i] - output_activations[idx * OUTPUT_SIZE + i];
        }

        // Calculate hidden layer error
        double hidden_error[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double error = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                error += output_error[j] * hidden_output_weights[i * OUTPUT_SIZE + j];
            }
            hidden_error[i] = error * sigmoid_derivative(hidden_activations[idx * HIDDEN_SIZE + i]);
        }

        // Update hidden to output weights
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                hidden_output_weights[i * OUTPUT_SIZE + j] += learning_rate * output_error[j] * hidden_activations[idx * HIDDEN_SIZE + i];
            }
        }

        // Update input to hidden weights
        for (int i = 0; i < image_size; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                input_hidden_weights[i * HIDDEN_SIZE + j] += learning_rate * hidden_error[j] * input_images[idx * image_size + i];
            }
        }
    }
}

// Function to calculate loss
double calculate_loss(double *output_activations, double *target_labels, int num_images) {
    double total_loss = 0.0;
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            total_loss -= target_labels[i * OUTPUT_SIZE + j] * log(output_activations[i * OUTPUT_SIZE + j]); // Cross-entropy loss
        }
    }
    return total_loss / num_images;
}

// Function to read IDX3 format image file
void read_idx3_file(const char *filename, uint8_t ***images, int *num_images, int *image_size) {
    // Function implementation remains the same as in the original code
}

// Function to read IDX1 format label file
void read_idx1_file(const char *filename, uint8_t **labels, int *num_labels) {
    // Function implementation remains the same as in the original code
}

// Function to normalize pixel values of images
void normalize_images(uint8_t **images, int num_images, int image_size) {
    // Function implementation remains the same as in the original code
}

// Function to flatten 2D images to 1D vectors
void flatten_images(uint8_t **images, int num_images, int image_size, double ***flattened_images) {
    // Function implementation remains the same as in the original code
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


    // Start timing
    clock_t start = clock();
 
    // Allocate memory for storing flattened training images, labels, and output activations on the GPU
    double *d_input_images, *d_training_labels, *d_input_hidden_weights, *d_hidden_output_weights,
           *d_hidden_activations, *d_output_activations;
    cudaMalloc((void **)&d_input_images, num_training_images * image_size * sizeof(double));
    cudaMalloc((void **)&d_training_labels, num_training_images * OUTPUT_SIZE * sizeof(double));
    cudaMalloc((void **)&d_input_hidden_weights, image_size * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void **)&d_hidden_output_weights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));
    cudaMalloc((void **)&d_hidden_activations, num_training_images * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void **)&d_output_activations, num_training_images * OUTPUT_SIZE * sizeof(double));

    // Transfer data from CPU to GPU memory
    cudaMemcpy(d_input_images, flattened_training_images[0], num_training_images * image_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_training_labels, training_labels, num_training_images * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for CUDA kernel launch
    int numThreadsPerBlock = 256;
    int numBlocks = (num_training_images + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Define weights
    double *input_hidden_weights, *hidden_output_weights;
    input_hidden_weights = (double *)malloc(image_size * HIDDEN_SIZE * sizeof(double));
    hidden_output_weights = (double *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));

    // Initialize weights randomly
    srand(time(NULL)); // Seed random number generator
    for (int i = 0; i < image_size * HIDDEN_SIZE; i++) {
        input_hidden_weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        hidden_output_weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    // Copy weights from CPU to GPU memory
    cudaMemcpy(d_input_hidden_weights, input_hidden_weights, image_size * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_output_weights, hidden_output_weights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Forward propagation
        forwardPropagationKernel<<<numBlocks, numThreadsPerBlock>>>(d_input_images, d_input_hidden_weights,
                                                                    d_hidden_output_weights, d_hidden_activations,
                                                                    d_output_activations, num_training_images, image_size);
        // Check for kernel launch errors
    cudaError_t forwardError = cudaGetLastError();
    if (forwardError != cudaSuccess) {
        printf("Forward propagation kernel launch failed: %s\n", cudaGetErrorString(forwardError));
        exit(EXIT_FAILURE);
    }

        // Synchronize GPU
        cudaDeviceSynchronize();

        // Calculate loss
        double loss = calculate_loss(d_output_activations, d_training_labels, num_training_images);
        printf("Epoch %d, Loss: %f\n", epoch + 1, loss);

        // Backward propagation
        backwardPropagationKernel<<<numBlocks, numThreadsPerBlock>>>(d_input_images, d_training_labels,
                                                                     d_hidden_activations, d_output_activations,
                                                                     d_input_hidden_weights, d_hidden_output_weights,
                                                                     num_training_images, image_size, LEARNING_RATE);
         // Check for kernel launch errors
    cudaError_t backwardError = cudaGetLastError();
    if (backwardError != cudaSuccess) {
        printf("Backward propagation kernel launch failed: %s\n", cudaGetErrorString(backwardError));
        exit(EXIT_FAILURE);
    }
    
            // Synchronize GPU
        cudaDeviceSynchronize();
    }

    // Allocate memory for storing output activations on the CPU
    double *output_activations_cpu = (double *)malloc(num_test_images * OUTPUT_SIZE * sizeof(double));

    // Copy output activations from GPU to CPU memory
    cudaMemcpy(output_activations_cpu, d_output_activations, num_test_images * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    // End timing
    clock_t end = clock();
    double elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Training time: %lf seconds\n", elapsed_time);

    // Evaluate the trained model on the test dataset
    int correct_predictions = 0;
    for (int i = 0; i < num_test_images; i++) {
        // Find the index of the output with the highest probability
        int predicted_label = 0;
        double max_prob = output_activations_cpu[i * OUTPUT_SIZE];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output_activations_cpu[i * OUTPUT_SIZE + j] > max_prob) {
                max_prob = output_activations_cpu[i * OUTPUT_SIZE + j];
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

    // Free CPU and GPU memory
    cudaFree(d_input_images);
    cudaFree(d_training_labels);
    cudaFree(d_input_hidden_weights);
    cudaFree(d_hidden_output_weights);
    cudaFree(d_hidden_activations);
    cudaFree(d_output_activations);
    free(input_hidden_weights);
    free(hidden_output_weights);
    free(output_activations_cpu);

    // Free memory allocated for training and test images and labels
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
