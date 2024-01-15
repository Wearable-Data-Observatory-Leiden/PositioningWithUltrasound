#include <Arduino_BuiltIn.h>
#include "NewPing.h"

#define TRIGGER_PIN  5
#define MAX_DISTANCE 400
#define NUM_SENSORS 1

// Pins for each sensor
const int echoPins[NUM_SENSORS] = {6};

// Create an array to hold the NewPing sensors
NewPing sensors[NUM_SENSORS] = {
  NewPing(TRIGGER_PIN, echoPins[0], MAX_DISTANCE)
};

// Arrays to store duration for each sensor
float durations[NUM_SENSORS];
// Number of iterations for median filter
const int iterations = 5;

void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    int sensorNumber = command - '0';  // Convert char to integer
    if (sensorNumber >= 1 && sensorNumber <= NUM_SENSORS) {
      // Measure duration for the specified sensor
      durations[sensorNumber - 1] = sensors[sensorNumber - 1].ping_median(iterations);
      Serial.print(sensorNumber);
      Serial.print(": ");
      Serial.print(durations[sensorNumber - 1]);
      Serial.println();
    } else {  // probably a higher number, this is just to cause the trigger
      durations[0] = sensors[0].ping_median(iterations); 
    }
  }
}