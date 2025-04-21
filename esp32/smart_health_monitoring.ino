#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"
#include "Protocentral_MAX30205.h" 

MAX30205 tempSensor;
MAX30105 particleSensor;
float weight=18.6;
float height=6.3;
const byte RATE_SIZE = 10; //Increase this for more averaging. 4 is good.
byte rates[RATE_SIZE]; //Array of heart rates
byte rateSpot = 0;
long lastBeat = 0; //Time at which the last beat occurred
const bool fahrenheittemp = false; // I'm showing the temperature in Fahrenheit, If you want to show the temperature in Celsius the make this variable false.
float beatsPerMinute;
int beatAvg;
MAX30105 particleSensor;

#define MAX_BRIGHTNESS 255

#if defined(__AVR_ATmega328P__) || defined(__AVR_ATmega168__)
//Arduino Uno doesn't have enough SRAM to store 100 samples of IR led data and red led data in 32-bit format
//To solve this problem, 16-bit MSB of the sampled data will be truncated. Samples become 16-bit data.
uint16_t irBuffer[100]; //infrared LED sensor data
uint16_t redBuffer[100];  //red LED sensor data
#else
uint32_t irBuffer[100]; //infrared LED sensor data
uint32_t redBuffer[100];  //red LED sensor data
#endif

int32_t bufferLength; //data length
int32_t spo2; //SPO2 value
int8_t validSPO2; //indicator to show if the SPO2 calculation is valid
int32_t heartRate; //heart rate value
int8_t validHeartRate; //indicator to show if the heart rate calculation is valid

byte pulseLED = 11; //Must be on PWM pin
byte readLED = 13; //Blinks with each data read

// MACRO Definitions
#define BP_START_PIN (2)         // start button of the blood pressure monitor device. replaced a transistor.   
#define VALVE_PIN (5)            // checks if the measurement is done.             
#define MEASURE_BEGIN_PIN (4)    // indicates that a measurement should start. this can be connected to switch or another MCU or raspberry pi.
#define PUMP_PIN (6)             // checks if the pump starts putting pressure on the calf.

volatile byte i2c_data_rx;       // indicates there are available data from the i2c bus.
volatile uint16_t count;         // indicates the total number of data collected.
volatile uint8_t sys, dia, hr;   // stored the measure values: systolic, diastolic and heart rate.
uint8_t bp_measure_done = 0;

unsigned long previousMillis = 0;
const long interval = 60000; // 60 second interval

void calculateDerivedHRV() {
  float rr[RATE_SIZE];
  float sum_sq_diff = 0.0;
  int valid_values = 0;

  // Convert BPM to RR intervals (in seconds)
  for (int i = 0; i < RATE_SIZE; i++) {
    if (rates[i] >= 20 && rates[i] <= 255) {
      rr[i] = 60.0 / rates[i];
      valid_values++;
    } else {
      rr[i] = 0;
    }
  }

  if (valid_values >= 2) {
    // Compute RMSSD
    for (int i = 0; i < RATE_SIZE - 1; i++) {
      if (rr[i] > 0 && rr[i+1] > 0) {
        float diff = rr[i+1] - rr[i];
        sum_sq_diff += diff * diff;
      }
    }

    float derivedHRV = sqrt(sum_sq_diff / (valid_values - 1));
    Serial.print(", Derived_HRV=");
    Serial.println(derivedHRV, 4); // print with 4 decimals
  }
}

// Function to calculate BMI
float calculateBMI(float weight_kg, float height_cm) {
    float height_m = height_cm / 100.0;  // convert cm to meters
    if (height_m <= 0) {
        return -1;  // error case
    }
    float bmi = weight_kg / (height_m * height_m);
    return bmi;
}

void setup()
{
  Serial.begin(115200);
  Serial.println("Initializing...");
    pinMode(pulseLED, OUTPUT);
  pinMode(readLED, OUTPUT);

  // Initialize sensor
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) //Use default I2C port, 400kHz speed
  {
    Serial.println("MAX30102 was not found. Please check wiring/power. ");
    while (1);
  }
  Serial.println("Place your index finger on the sensor with steady pressure.");
  Serial.println(F("Attach sensor to finger with rubber band. Press any key to start conversion"));
  while (Serial.available() == 0) ; //wait until user presses a key
  Serial.read();

  byte ledBrightness = 60; //Options: 0=Off to 255=50mA
  byte sampleAverage = 4; //Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 2; //Options: 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
  byte sampleRate = 100; //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
  int pulseWidth = 411; //Options: 69, 118, 215, 411
  int adcRange = 4096; //Options: 2048, 4096, 8192, 16384

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange); 
  particleSensor.setup(); //Configure sensor with default settings
  particleSensor.setPulseAmplitudeRed(0x0A); //Turn Red LED to low to indicate sensor is running
  particleSensor.setPulseAmplitudeGreen(0); //Turn off Green LED
  tempSensor.begin(); 
  Serial.println("temperature sensor MAX30205 is intailsed");
   pinMode(BP_START_PIN, OUTPUT);
  pinMode(VALVE_PIN, INPUT);
  pinMode(MEASURE_BEGIN_PIN, INPUT_PULLUP);
  pinMode(PUMP_PIN, INPUT);
  Wire.begin(0x50);                           // the address of the EEPROM is 0x50. The arduino should be the same.
  Wire.onReceive(receiveEvent);               // this is the interrupt initialization for the i2c data
}

void loop() {
  unsigned long currentMillis = millis();
  
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    
    // Print all sensor values at 60-second interval
    Serial.println("===== 60-Second Sensor Readings =====");
    
    // Heart rate data
    Serial.print("Heart Rate - BPM=");
    Serial.print(beatsPerMinute);
    Serial.print(", Avg BPM=");
    Serial.println(beatAvg);
    
    // SpO2 data
    Serial.print("SPO2=");
    Serial.print(spo2);
    Serial.print(", SPO2Valid=");
    Serial.println(validSPO2);
    
    // Temperature data
    float temp = tempSensor.getTemperature();
    Serial.print("Body Temperature: ");
    Serial.println(temp);
    
    // Respiratory rate
    int respiratory_rate = analogRead(A0);
    Serial.print("Respiratory Rate: ");
    Serial.println(respiratory_rate);
    
    // Blood pressure data (if available)
    if (count >= 28) {
      Serial.print("Blood Pressure - Systolic: ");
      Serial.print(sys);
      Serial.print(", Diastolic: ");
      Serial.println(dia);
    }
    
    // Derived metrics
    calculateDerivedHRV();
    if (sys > 0 && dia > 0) {
      int derivedPulsePressure = sys - dia;
      float map = dia + (sys - dia) / 3.0;
      Serial.print("Derived Pulse Pressure: ");
      Serial.println(derivedPulsePressure);
      Serial.print("Derived MAP: ");
      Serial.println(map);
    }
    
    // BMI
    float bmi = calculateBMI(weight, height);
    Serial.print("BMI: ");
    Serial.println(bmi);
    
    Serial.println("====================================");
  }

  // Original sensor reading code continues to run in background
  long irValue = particleSensor.getIR();

  if (checkForBeat(irValue) == true)
  {
    //We sensed a beat!
    long delta = millis() - lastBeat;
    lastBeat = millis();

    beatsPerMinute = 60 / (delta / 1000.0);

    if (beatsPerMinute < 255 && beatsPerMinute > 20)
    {
      rates[rateSpot++] = (byte)beatsPerMinute; //Store this reading in the array
      rateSpot %= RATE_SIZE; //Wrap variable

      //Take average of readings
      beatAvg = 0;
      for (byte x = 0 ; x < RATE_SIZE ; x++)
        beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }

  if (irValue < 50000)
    Serial.print(" No finger?");

  bufferLength = 100; //buffer length of 100 stores 4 seconds of samples running at 25sps

  //read the first 100 samples, and determine the signal range
  for (byte i = 0 ; i < bufferLength ; i++)
  {
    while (particleSensor.available() == false) //do we have new data?
      particleSensor.check(); //Check the sensor for new data

    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample(); //We're finished with this sample so move to next sample
  }

  //calculate heart rate and SpO2 after first 100 samples (first 4 seconds of samples)
  maxim_heart_rate_and_oxygen_saturation(irBuffer, bufferLength, redBuffer, &spo2, &validSPO2, &heartRate, &validHeartRate);

  //Continuously taking samples from MAX30102.  Heart rate and SpO2 are calculated every 1 second
  while (1)
  {
    //dumping the first 25 sets of samples in the memory and shift the last 75 sets of samples to the top
    for (byte i = 25; i < 100; i++)
    {
      redBuffer[i - 25] = redBuffer[i];
      irBuffer[i - 25] = irBuffer[i];
    }

    //take 25 sets of samples before calculating the heart rate.
    for (byte i = 75; i < 100; i++)
    {
      while (particleSensor.available() == false) //do we have new data?
        particleSensor.check(); //Check the sensor for new data

      digitalWrite(readLED, !digitalRead(readLED)); //Blink onboard LED with every data read

      redBuffer[i] = particleSensor.getRed();
      irBuffer[i] = particleSensor.getIR();
      particleSensor.nextSample(); //We're finished with this sample so move to next sample
    }

    //After gathering 25 new samples recalculate HR and SP02
    maxim_heart_rate_and_oxygen_saturation(irBuffer, bufferLength, redBuffer, &spo2, &validSPO2, &heartRate, &validHeartRate);
    delay(1);
    
    if (digitalRead(MEASURE_BEGIN_PIN) == 0)        // The arduino is instructed to start the measurement.
    {
      digitalWrite(BP_START_PIN, HIGH);             // Emulating a push on the button.
      delay(200);
      digitalWrite(BP_START_PIN, LOW);

      delay(1000);
      Serial.println("Attemp to start pump...");
      delay(2000);

      while (digitalRead(PUMP_PIN) == 1)
      {
        digitalWrite(BP_START_PIN, HIGH);             // Emulating a push on the button.
        delay(200);
        digitalWrite(BP_START_PIN, LOW);

        delay(1000);
        Serial.println("Attemp to start pump...");
        delay(2000);
      }

      Serial.println("Pump now started...");

      delay(5000);

      //need to secure that the value is already closed.
      while (digitalRead(VALVE_PIN) == 0)
      {
        bp_measure_done = 0;
        Serial.println("wait...");
        delay(2000);
      }

      delay(2000);

      Serial.println("Done reading...");

      digitalWrite(BP_START_PIN, HIGH);
      delay(200);
      digitalWrite(BP_START_PIN, LOW);

      delay(500);

      if (count == 0)
      {
        Serial.print("<");
        Serial.print('0');
        Serial.print(",");
        Serial.print('0');
        Serial.println(">");
      }
      else if (count == 35)
      {
        Serial.print("<");
        Serial.print(sys);
        Serial.print(",");
        Serial.print(dia);
        Serial.println(">");
      }
      else
      {
        Serial.print("<");
        Serial.print('1');
        Serial.print(",");
        Serial.print('1');
        Serial.println(">");
      }
      count = 0;
    }
  }
}

void receiveEvent(int iData)   // Interrupt service routine.
{
  if ( iData > 0 )
  {
    while ( iData-- )
    {
      i2c_data_rx = Wire.read();
      count++;

      if (count == 28)
      {
        sys = i2c_data_rx;// syslotiolic blood pressure
        Serial.println("systolic blood pressure");
         Serial.println(sys);
      }
      if (count == 29)
      {
        dia = i2c_data_rx;//diaphoric blood presuure
        Serial.println("diastolic blood pressure");
         Serial.println(dia);
      }
      if (count == 30)
      {
        hr = i2c_data_rx;
      }
      if (sys > 0 && dia > 0)
      {
        int derivedPulsePressure = sys - dia;
        float map = dia + (sys - dia) / 3.0;
        Serial.print("Derived_Pulse_Pressure=");
        Serial.println(derivedPulsePressure);
        Serial.print("Derived_MAP:");
        Serial.println(map);
      }
    }
  }
}