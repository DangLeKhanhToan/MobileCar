#include "auto_velocity.h"
#include "motor.h"
#include "config.h"
#include "calibration.h"
#include <Arduino.h>
#include <math.h>

void velocityCommand(float linearMps, float angularRadS) {
  // Loaded skid-steer bases need a dedicated turn profile; kinematic PWM is
  // otherwise below static-friction torque for an in-place turn.
  if (fabs(linearMps) < 0.001f && fabs(angularRadS) > 0.001f) {
    if (angularRadS > 0) setMotorTargets(-TURN_LEFT_PWM, TURN_RIGHT_PWM);
    else setMotorTargets(TURN_LEFT_PWM, -TURN_RIGHT_PWM);
    return;
  }
  const float leftMps = linearMps - angularRadS * WHEEL_SEPARATION_M * 0.5f;
  const float rightMps = linearMps + angularRadS * WHEEL_SEPARATION_M * 0.5f;
  const float maxWheelMps = WHEEL_RADIUS_M * MAX_WHEEL_ANGULAR_RAD_S;

  float leftPwm = leftMps * MAX_PWM / maxWheelMps;
  float rightPwm = rightMps * MAX_PWM / maxWheelMps;

  // Preserve curvature when either wheel request exceeds the available PWM.
  float largest = max(fabs(leftPwm), fabs(rightPwm));
  if (largest > MAX_PWM) {
    leftPwm *= MAX_PWM / largest;
    rightPwm *= MAX_PWM / largest;
  }
  setMotorTargets((int)leftPwm, (int)rightPwm);
}
