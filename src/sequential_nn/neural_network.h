#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define INPUT_SIZE 2
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.1

void forward(double input[], double weights_input_hidden[][INPUT_SIZE], 
             double hidden[], double weights_hidden_output[][HIDDEN_SIZE],
             double output[]);
void backward(double input[], double target[], double output[], 
              double hidden[], double weights_hidden_output[][HIDDEN_SIZE], 
              double weights_input_hidden[][INPUT_SIZE]);

#endif /* NEURAL_NETWORK_H */
