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
  // Measure duration for all sensors
  for (int i = 0; i < NUM_SENSORS; i++) {
    durations[i] = sensors[i].ping_median(iterations);
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.print(durations[i]);
    Serial.println();
    delay(50);
  }
}