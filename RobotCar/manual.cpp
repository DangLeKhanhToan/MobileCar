#include "manual.h"
#include "motor.h"
#include "config.h"
#include "calibration.h"

void manualCommand(char cmd) {
  switch (cmd) {
    case 'F': setMotorTargets(MANUAL_PWM, MANUAL_PWM, true); break;
    case 'B': setMotorTargets(-MANUAL_PWM, -MANUAL_PWM, true); break;
    case 'L': setMotorTargets(-MANUAL_PWM, MANUAL_PWM, true); break;
    case 'R': setMotorTargets(MANUAL_PWM, -MANUAL_PWM, true); break;
    case 'S': stopMotor(); break;
  }
}
