#include <Wire.h>
#include "I2Cdev.h"
#include "MPU6050.h"

#include <Servo.h>

Servo myservo;  
Servo myservo2;

#define LEN 250
#define LIM1 10000L

MPU6050 accelgyro;

int16_t ax, ay, az;
int16_t gx, gy, gz;

int16_t timeA, time_rx,t_delta;

uint8_t i=0, i_rx, i_delta;

int16_t timeB,timeB_old,tmp1;

char buf1[LEN];

void setup() {

  myservo.attach(7);
  myservo2.attach(8);
    
  Wire.begin();
  
  Serial1.begin(38400);
  
  accelgyro.initialize();

  accelgyro.setFullScaleGyroRange(MPU6050_GYRO_FS_1000);

  timeB = timeB_old = (int16_t) (millis() % LIM1);
}

void loop() {

  do {

    delay(1);  
  
    tmp1 = timeB = (int16_t)(millis() % LIM1);
    if(timeB - timeB_old < 0) timeB += LIM1;

  } while(timeB - timeB_old < 100); 

  timeB_old = tmp1;
  
  i++;

  accelgyro.getAcceleration(&ax, &ay, &az);
  accelgyro.getRotation(&gx, &gy, &gz);

  char str1[LEN];  
  
  timeA = (int16_t)(millis()%1000);

  t_delta = timeA - time_rx;

  if(t_delta < 0) t_delta += 1000;

  i_delta = i - i_rx; 

  sprintf(str1,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", ax,ay,az,gx,gy,gz,i,t_delta, i_delta, timeA);

  Serial1.println(str1);

  int c;
  int cnt;

  cnt = 0;
  do {

    if(Serial1.available()) {
      c = Serial1.read();

      buf1[cnt++] = c;
    }
  } while(c != '\n' && c != '\r' && cnt < (LEN -1) );

  buf1[cnt] ='\0';

  char *str2 = strtok(buf1, ":");

  i_rx = (uint8_t) atoi(str2);

  str2 = strtok(NULL, ":");

  time_rx = (int16_t) atoi(str2);

  str2 = strtok(NULL, ":");

  int16_t val;

  val = (int16_t) atoi(str2);

  int16_t val1,val2;

  val1 = map(val, 0, 1023, 10, 100);     
  val2 = map(val, 0, 1023, 175, 85);
  
  myservo.write(val1); 
  delay(2);                   
  myservo2.write(val2);
  delay(2);
}
