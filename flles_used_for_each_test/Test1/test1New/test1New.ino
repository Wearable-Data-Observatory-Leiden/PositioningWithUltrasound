#include <Arduino_BuiltIn.h>
#include "NewPing.h"

#define TRIGGER_PIN  5
#define MAX_DISTANCE 400
#define NUM_SENSORS 2

// Pins for each sensor
const int echoPins[NUM_SENSORS] = {6, 7};

// Create an array to hold the NewPing sensors
NewPing sensors[NUM_SENSORS] = {
  NewPing(TRIGGER_PIN, echoPins[0], MAX_DISTANCE),
  NewPing(TRIGGER_PIN, echoPins[1], MAX_DISTANCE)
};

// Arrays to store duration for each sensor
float durations[NUM_SENSORS];

void setup() {
  Serial.begin(115200);
}

void loop() {
  // Measure duration for all sensors
  for (int i = 0; i < NUM_SENSORS; i++) {
    durations[i] = sensors[i].ping();
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.print(durations[i]);
    Serial.println();
    delay(50);
  }
}