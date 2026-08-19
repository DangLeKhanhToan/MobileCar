#ifndef MOTOR_H
#define MOTOR_H

void motorInit();
void setMotorTargets(int leftPwm, int rightPwm, bool manualMode = false);
void setRawMotorTargets(int leftPwm, int rightPwm);
void updateMotors(unsigned long nowMs);
void stopMotor();
void hardStopMotor();

#endif
