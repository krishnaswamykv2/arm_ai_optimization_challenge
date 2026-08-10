/*
 * EcoSphere -- STM32U585 sketch (real-time side of the UNO Q)
 *
 * Responsibilities:
 *   1. Read the three hazard sensors (gas, tilt, vibration)
 *   2. Expose those readings to the QRB2210/Python side via Bridge RPC
 *   3. Receive the classified hazard state from Python
 *   4. Actuate a physical response (LED / buzzer / motor-stop)
 *
 * ============================================================
 * ONLY SECTION THAT NEEDS FILLING IN ONCE HARDWARE IS AVAILABLE:
 * see "HARDWARE CONFIG" block below. Everything else is complete
 * as-is and doesn't need to change based on your specific wiring.
 * ============================================================
 *
 * NOT YET TESTED ON HARDWARE. Written against the documented
 * Bridge.provide() / Bridge.provide_safe() pattern (see
 * HARDWARE_REQUIREMENTS.md for sources) but not compiled or run,
 * since no board is available right now. Compile-check this before
 * trusting it on the actual rover.
 */

#include <Arduino_RouterBridge.h>

// ============================================================
// HARDWARE CONFIG -- FILL THIS IN ONCE YOU HAVE THE ROVER
// ============================================================
// Gas sensor: change pin, and confirm whether it's analog or digital/I2C.
// If analog (e.g. MQ-series), GAS_PIN is an analog pin, gas reading
// comes from analogRead(). If I2C, this needs different code -- flag
// that back before filling this in, since the reading logic differs.
#define GAS_SENSOR_IS_ANALOG true   // set false if it's I2C instead
#define GAS_PIN A0                  // TODO: confirm actual pin

// IMU (tilt + vibration): most common is I2C (e.g. MPU6050-style).
// If yours is different, flag it -- the read logic below assumes I2C.
#define IMU_I2C_ADDRESS 0x68        // TODO: confirm actual I2C address

// Actuator pins -- fill in whichever are actually wired.
#define LED_PIN 2                   // TODO: confirm actual pin
#define BUZZER_PIN 3                // TODO: confirm actual pin, or remove if not wired
#define MOTOR_STOP_PIN 4            // TODO: confirm actual pin, or remove if not wired
#define HAS_BUZZER true             // set false if no buzzer wired
#define HAS_MOTOR_CONTROL false     // set true once motor-stop is wired

// Calibration: raw sensor value -> real units (ppm / degrees / g).
// TODO: fill in once the actual sensor's datasheet/calibration curve
// is known. These are placeholders and MUST be replaced -- do not
// ship with these defaults.
#define GAS_ADC_TO_PPM_SCALE 1.0    // TODO: replace with real calibration
#define GAS_ADC_TO_PPM_OFFSET 0.0   // TODO: replace with real calibration
// ============================================================


// ---- Sensor reading functions (RPC-exposed to Python) ----

float get_gas_reading() {
#if GAS_SENSOR_IS_ANALOG
  int raw = analogRead(GAS_PIN);
  return (raw * GAS_ADC_TO_PPM_SCALE) + GAS_ADC_TO_PPM_OFFSET;
#else
  // TODO: I2C gas sensor read logic goes here if GAS_SENSOR_IS_ANALOG is false
  return 0.0;
#endif
}

float get_tilt_reading() {
  // TODO: read from IMU at IMU_I2C_ADDRESS, convert to degrees.
  // Placeholder returns 0 until IMU wiring is confirmed.
  return 0.0;
}

float get_vibration_reading() {
  // TODO: read from IMU at IMU_I2C_ADDRESS, convert to g.
  // Placeholder returns 0 until IMU wiring is confirmed.
  return 0.0;
}


// ---- Hazard response (RPC-exposed to Python, called with the result) ----

void set_hazard_response(String hazard_state) {
  if (hazard_state == "SAFE") {
    digitalWrite(LED_PIN, LOW);
#if HAS_BUZZER
    noTone(BUZZER_PIN);
#endif
#if HAS_MOTOR_CONTROL
    digitalWrite(MOTOR_STOP_PIN, LOW);  // motors free to run
#endif
  } else if (hazard_state == "ELEVATED") {
    digitalWrite(LED_PIN, HIGH);  // solid on
#if HAS_BUZZER
    noTone(BUZZER_PIN);
#endif
#if HAS_MOTOR_CONTROL
    digitalWrite(MOTOR_STOP_PIN, LOW);
#endif
  } else if (hazard_state == "CRITICAL") {
    // Blink pattern for critical -- handled in loop() via blink state,
    // not blocking here, since blocking would stall Bridge RPC handling.
    critical_active = true;
#if HAS_BUZZER
    tone(BUZZER_PIN, 1000);  // steady alarm tone
#endif
#if HAS_MOTOR_CONTROL
    digitalWrite(MOTOR_STOP_PIN, HIGH);  // stop motors immediately
#endif
    return;
  }
  critical_active = false;
}


// ---- Setup / loop ----

bool critical_active = false;
unsigned long last_blink = 0;
bool led_blink_state = false;

void setup() {
  pinMode(LED_PIN, OUTPUT);
#if HAS_BUZZER
  pinMode(BUZZER_PIN, OUTPUT);
#endif
#if HAS_MOTOR_CONTROL
  pinMode(MOTOR_STOP_PIN, OUTPUT);
#endif

  Bridge.begin();
  Bridge.provide_safe("get_gas_reading", get_gas_reading);
  Bridge.provide_safe("get_tilt_reading", get_tilt_reading);
  Bridge.provide_safe("get_vibration_reading", get_vibration_reading);
  Bridge.provide_safe("set_hazard_response", set_hazard_response);
}

void loop() {
  // Non-blocking blink for CRITICAL state -- keeps Bridge responsive.
  if (critical_active) {
    unsigned long now = millis();
    if (now - last_blink > 200) {
      led_blink_state = !led_blink_state;
      digitalWrite(LED_PIN, led_blink_state ? HIGH : LOW);
      last_blink = now;
    }
  }
}
