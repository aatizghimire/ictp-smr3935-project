#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define INPUT_SIZE 2
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 1

//Sigmoid Activation Function
double sigmoid(double x){
    return 1.0/(1.0 + exp(-x)); 
}

// Derivative of the sigmoid function 
double derivative_sigmoid(double x){
    return x * (1.0 - x);
}

//forward propagation
void forward_propagation(double input[INPUT_SIZE], double weight_input_hidden[HIDDEN_SIZE][INPUT_SIZE], double hidden[HIDDEN_SIZE], double weight_hidden_output[OUTPUT_SIZE], double output[OUTPUT_SIZE]){
    
    //calculate hidden layer
    for (int i=0; i <HIDDEN_SIZE ; i++){
        double sum=0;
        for(int j=0; j<INPUT_SIZE; j++){
            sum +=input[i] * weight_input_hidden[i][j];
        }
        hidden[i]=sigmoid(sum);
        printf("Hidden value %lf",hidden[i]);
    }

    //calculate output layer
    for (int i=0; i <OUTPUT_SIZE ; i++){
        double sum=0;
        for(int j=0; j<HIDDEN_SIZE; j++){
            sum +=hidden[i] * weight_hidden_output[i][j];
        }
        output[i]=sigmoid(sum);
        printf("Hidden value %lf",hidden[i]);
    }
}


void main(){

    void forward_propagation();

}