#include <Arduino_BuiltIn.h>
#include "NewPing.h"

#define TRIGGER_PIN     5
#define ECHO_PIN        6
#define MAX_DISTANCE    400

NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE); // NewPing setup of pins and maximum distance.

void setup() {
  Serial.begin(115200); // Open serial monitor at 115200 baud to see ping results.
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    Serial.print("Ping: ");
    Serial.print(sonar.ping_cm()); // Send ping, get distance in cm and print result (0 = outside set distance range)
    Serial.println("cm");
  }
}