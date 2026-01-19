#include <Adafruit_NeoPixel.h>

#define LED_PIN 6
#define LED_COUNT 61   // 24 + 16 + 12 + 8 + 1

Adafruit_NeoPixel strip(
  LED_COUNT,
  LED_PIN,
  NEO_GRB + NEO_KHZ800
);

uint8_t brightness = 250;
uint32_t currentColor;

void setAllSolid(uint32_t color) {
  for (int i = 0; i < LED_COUNT; i++) {
    strip.setPixelColor(i, color);
  }
  strip.show();
}

void clearSerial() {
  while(Serial.available()) {
    Serial.read();
  }
}


void setup() {
  Serial.begin(9600);

  strip.begin();
  strip.setBrightness(brightness);

  currentColor = strip.Color(255, 255, 255); // default white
  setAllSolid(currentColor);

  Serial.println("Commands: w r b y  |  = +bright  - dim");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    switch (c) {
      case 'w':
        currentColor = strip.Color(255, 255, 255);
        break;
    
      case 'r':
        currentColor = strip.Color(255, 0, 0);
        break;

      case 'b':
        currentColor = strip.Color(0, 0, 255);
        break;

      case 'y':
        currentColor = strip.Color(255, 255, 0);
        break;

      case '=':
        if (brightness < 245) brightness += 10;
        strip.setBrightness(brightness);
        break;

      case '-':
        if (brightness > 10) brightness -= 10;
        strip.setBrightness(brightness);
        break;

      default:
        return; // ignore anything else
    }

    setAllSolid(currentColor);

    Serial.print("Brightness: ");
    Serial.println(brightness);
    clearSerial();
    delay(200);
  }
}
