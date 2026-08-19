#include "safety.h"
#include "motor.h"

bool emergencyStop = false;

void emergency() {
  emergencyStop = true;
  hardStopMotor();
}

void resetEmergency() {
  emergencyStop = false;
  hardStopMotor();
}
