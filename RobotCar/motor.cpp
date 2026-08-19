#include "motor.h"
#include "config.h"
#include "calibration.h"
#include <Arduino.h>

static float currentLeft = 0.0f, currentRight = 0.0f;
static int targetLeft = 0, targetRight = 0;
static bool rawOutput = false, manualRamp = false;
static unsigned long lastUpdateMs = 0, kickUntilMs = 0;

static void writeMotor(uint8_t pinA, uint8_t pinB, int pwm) {
  pwm = constrain(pwm, -MAX_PWM, MAX_PWM);
  if (pwm >= 0) { analogWrite(pinA, pwm); analogWrite(pinB, 0); }
  else { analogWrite(pinA, 0); analogWrite(pinB, -pwm); }
}

static int calibratedPwm(int requested, bool left) {
  if (!requested) return 0;
  bool forward = requested > 0;
  int minimum;
  float gain;
  if (left) {
    minimum = forward ? LEFT_MIN_FORWARD_PWM : LEFT_MIN_REVERSE_PWM;
    gain = forward ? LEFT_FORWARD_GAIN : LEFT_REVERSE_GAIN;
  } else {
    minimum = forward ? RIGHT_MIN_FORWARD_PWM : RIGHT_MIN_REVERSE_PWM;
    gain = forward ? RIGHT_FORWARD_GAIN : RIGHT_REVERSE_GAIN;
  }
  int output = minimum + (long)abs(requested) * (MAX_PWM - minimum) / MAX_PWM;
  output = constrain((int)(output * gain), 0, MAX_PWM);
  return forward ? output : -output;
}

static float approach(float value, float target, float amount) {
  if (value < target) return min(value + amount, target);
  if (value > target) return max(value - amount, target);
  return value;
}

void motorInit() {
  pinMode(LEFT_PWM_A, OUTPUT); pinMode(LEFT_PWM_B, OUTPUT);
  pinMode(RIGHT_PWM_A, OUTPUT); pinMode(RIGHT_PWM_B, OUTPUT);
  hardStopMotor(); lastUpdateMs = millis();
}

void setMotorTargets(int leftPwm, int rightPwm, bool isManual) {
  rawOutput = false; manualRamp = isManual;
  int newLeft = calibratedPwm(constrain(leftPwm, -MAX_PWM, MAX_PWM), true);
  int newRight = calibratedPwm(constrain(rightPwm, -MAX_PWM, MAX_PWM), false);
  if (!targetLeft && !targetRight && (newLeft || newRight)) kickUntilMs = millis() + STARTUP_KICK_MS;
  targetLeft = newLeft; targetRight = newRight;
}

void setRawMotorTargets(int leftPwm, int rightPwm) {
  rawOutput = true; manualRamp = true; kickUntilMs = 0;
  targetLeft = constrain(leftPwm, -MAX_PWM, MAX_PWM);
  targetRight = constrain(rightPwm, -MAX_PWM, MAX_PWM);
}

void updateMotors(unsigned long nowMs) {
  unsigned long elapsed = nowMs - lastUpdateMs;
  if (elapsed < CONTROL_PERIOD_MS) return;
  lastUpdateMs = nowMs;
  float accel = manualRamp ? MANUAL_ACCEL_PWM_PER_SEC : AUTO_ACCEL_PWM_PER_SEC;
  float decel = manualRamp ? MANUAL_DECEL_PWM_PER_SEC : AUTO_DECEL_PWM_PER_SEC;
  currentLeft = approach(currentLeft, targetLeft,
      (abs(targetLeft) > abs((int)currentLeft) ? accel : decel) * elapsed / 1000.0f);
  currentRight = approach(currentRight, targetRight,
      (abs(targetRight) > abs((int)currentRight) ? accel : decel) * elapsed / 1000.0f);
  int leftOut = (int)currentLeft, rightOut = (int)currentRight;
  if (!rawOutput && (long)(kickUntilMs - nowMs) > 0) {
    if (targetLeft) leftOut = targetLeft > 0 ? max(leftOut, STARTUP_KICK_PWM) : min(leftOut, -STARTUP_KICK_PWM);
    if (targetRight) rightOut = targetRight > 0 ? max(rightOut, STARTUP_KICK_PWM) : min(rightOut, -STARTUP_KICK_PWM);
  }
  writeMotor(LEFT_PWM_A, LEFT_PWM_B, leftOut);
  writeMotor(RIGHT_PWM_A, RIGHT_PWM_B, rightOut);
}

void stopMotor() { targetLeft = targetRight = 0; kickUntilMs = 0; }

void hardStopMotor() {
  targetLeft = targetRight = 0; currentLeft = currentRight = 0.0f; kickUntilMs = 0;
  writeMotor(LEFT_PWM_A, LEFT_PWM_B, 0); writeMotor(RIGHT_PWM_A, RIGHT_PWM_B, 0);
}
