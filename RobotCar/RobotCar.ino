#include <SoftwareSerial.h>
#include "config.h"
#include "motor.h"
#include "safety.h"

// Fixed wiring: HC-05 remains on D2/D3. USB Serial is also accepted so the
// same protocol can later run from ROS on Jetson Orin Nano.
SoftwareSerial BT(BT_RX_PIN, BT_TX_PIN);

char motionState = 'S';
int vSpeed = 155;
unsigned long lastControlMs = 0;

static void applyMotion() {
  if (emergencyStop || motionState == 'S' || vSpeed == 0) {
    hardStopMotor();
    return;
  }

  // Inner wheel remains positive during a moving turn. This lets the base
  // turn while translating instead of pivoting in place.
  const int inner = (long)vSpeed * 45 / 100;

  if (motionState == 'F')      setRawMotorTargets( vSpeed,  vSpeed);
  else if (motionState == 'B') setRawMotorTargets(-vSpeed, -vSpeed);
  else if (motionState == 'L') setRawMotorTargets(-vSpeed,  vSpeed);
  else if (motionState == 'R') setRawMotorTargets( vSpeed, -vSpeed);
  else if (motionState == 'G') setRawMotorTargets( inner,   vSpeed); // forward-left
  else if (motionState == 'I') setRawMotorTargets( vSpeed,  inner);  // forward-right
  else if (motionState == 'H') setRawMotorTargets(-inner,  -vSpeed); // reverse-left
  else if (motionState == 'J') setRawMotorTargets(-vSpeed, -inner);  // reverse-right
  else hardStopMotor();
}

// Discrete commands select a target speed level. motor.cpp ramps continuously
// from the current PWM to this target, so these are not instantaneous states.
static bool setSpeedLevel(char state) {
  if (state == '0')      vSpeed = 0;
  else if (state == '4') vSpeed = 100;
  else if (state == '6') vSpeed = 155;
  else if (state == '7') vSpeed = 180;
  else if (state == '8') vSpeed = 200;
  else if (state == '9') vSpeed = 230;
  else if (state == 'q') vSpeed = 255;
  else return false;
  applyMotion();
  return true;
}

static void acknowledge(Stream &source, char state) {
  source.print(F("ACK,"));
  source.print(state);
  source.print(',');
  source.print(vSpeed);
  source.print(',');
  source.println(motionState);
}

static void processCommand(char state, Stream &source) {
  if (state == '\r' || state == '\n' || state == ' ') return;

  if (state == 'E') {
    emergency();
    motionState = 'S';
    acknowledge(source, state);
    return;
  }
  if (state == 'X') {
    resetEmergency();
    motionState = 'S';
    lastControlMs = millis();
    acknowledge(source, state);
    return;
  }
  if (emergencyStop) return;

  if (state == 'K') { // heartbeat; deliberately no ACK
    lastControlMs = millis();
    return;
  }
  if (setSpeedLevel(state)) {
    lastControlMs = millis();
    acknowledge(source, state);
    return;
  }
  if (state == 'F' || state == 'B' || state == 'L' || state == 'R' ||
      state == 'G' || state == 'I' || state == 'H' || state == 'J' || state == 'S') {
    motionState = state;
    lastControlMs = millis();
    applyMotion();
    acknowledge(source, state);
  }
}

static void readCommands(Stream &port) {
  while (port.available()) processCommand((char)port.read(), port);
}

void setup() {
  Serial.begin(USB_BAUD); // preferred wired link to Jetson
  BT.begin(BT_BAUD);    // existing HC-05 remains at 9600
  motorInit();
  hardStopMotor();
  lastControlMs = millis();
  Serial.println(F("READY,SIMPLE_DRIVE,V1"));
  BT.println(F("READY,SIMPLE_DRIVE,V1"));
}

void loop() {
  const unsigned long now = millis();
  readCommands(Serial);
  readCommands(BT);
  if (!emergencyStop && now - lastControlMs > COMMAND_WATCHDOG_MS) {
    motionState = 'S';
    hardStopMotor();
  }
  updateMotors(now);
}
