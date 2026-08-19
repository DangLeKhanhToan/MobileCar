#ifndef CONFIG_H
#define CONFIG_H

// Fixed hardware wiring. Do not change without rewiring the robot.
#define BT_RX_PIN 2  // Arduino RX <- HC-05 TX
#define BT_TX_PIN 3  // Arduino TX -> HC-05 RX (use a voltage divider)
#define LEFT_PWM_A 9
#define LEFT_PWM_B 10
#define RIGHT_PWM_A 5
#define RIGHT_PWM_B 6

#define USB_BAUD 115200
#define BT_BAUD 9600

// Robot geometry (user-confirmed wheel radius).
#define WHEEL_RADIUS_M 0.03f
#define WHEEL_SEPARATION_M 0.13f

// Physical limits; robot-specific measurements live in calibration.h.
#define MAX_WHEEL_ANGULAR_RAD_S 20.0f
#define MAX_PWM 255

#define COMMAND_WATCHDOG_MS 500UL
#define CONTROL_PERIOD_MS 20UL
#define COMMAND_BUFFER_SIZE 64

#endif
