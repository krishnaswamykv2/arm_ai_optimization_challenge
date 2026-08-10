# What we need from the rover, exactly

This is the complete list of information/checks needed before
`uno_q_sketch/sketch.ino` can be finished and deployed. Nothing else
is blocked on hardware — everything else (Python side, model,
pipeline) is already built and working without needing any of this.

## 1. Gas sensor
- [ ] Which pin(s) is it wired to?
- [ ] Is it analog (e.g. MQ-series, reads via `analogRead()`) or
      digital/I2C?
- [ ] If analog: what's the sensor's datasheet calibration curve
      (raw ADC value -> ppm)? If you don't have this, note the
      sensor's exact model number and I'll look it up.
- [ ] Confirm the sensor is powered and giving *some* reading right
      now (even an uncalibrated one) — tells us it's wired correctly
      before we worry about calibration accuracy.

## 2. IMU (tilt + vibration)
- [ ] What's the exact IMU model (e.g. MPU6050, MPU9250, other)?
- [ ] Confirm it's I2C — if it's SPI or analog instead, say so, the
      sketch currently assumes I2C and needs different code otherwise.
- [ ] What's its I2C address (usually in the datasheet, often 0x68 or
      0x69 — but confirm, don't assume)?
- [ ] Confirm whether tilt (degrees) and vibration (g) both come from
      the same IMU, or are they separate sensors?

## 3. Actuator(s)
- [ ] What's actually wired right now — LED only? LED + buzzer? Any
      motor-stop control line?
- [ ] For each one wired: which pin?
- [ ] If motor-stop is wired: what signal actually stops the motors —
      a digital HIGH/LOW to a motor driver enable pin, cutting PWM to
      zero, or something else? This matters — get it wrong and
      "critical -> stop motors" won't actually work.

## 4. Sanity check once the above is filled in
- [ ] Compile the sketch in Arduino App Lab — does it build clean?
- [ ] Upload it, open the serial monitor, confirm sensor values print
      and look physically reasonable (e.g. gas reading changes when
      you breathe near the sensor, tilt changes when you tilt the
      board).
- [ ] Run `12_live_rover_pipeline.py` on the QRB2210 side, confirm it
      prints live classified readings without Bridge errors.
- [ ] Manually trigger each hazard tier (e.g. tilt the board past the
      critical threshold) and visually confirm the LED/buzzer/motor
      actually responds.

## What NOT to worry about
- Pin numbers for anything not listed above — the Python side never
  touches pin numbers, so nothing else needs this info.
- Getting calibration perfectly accurate on the first try — an
  uncalibrated-but-wired sensor is enough to prove the loop works
  end-to-end. Accuracy can be refined after the loop is proven.
