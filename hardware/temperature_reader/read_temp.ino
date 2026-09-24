#include "max6675.h"

// Cảm biến 1 (Phòng): SCK-D4, SO-D5, CS-D9
int thermo1_CLK = 4;
int thermo1_CS  = 9;
int thermo1_SO  = 5;
MAX6675 roomTemp(thermo1_CLK, thermo1_CS, thermo1_SO);

// Cảm biến 2 (Lõi): SCK-D13, SO-D12, CS-D10
int thermo2_CLK = 13;
int thermo2_CS  = 10;
int thermo2_SO  = 12;
MAX6675 coreTemp(thermo2_CLK, thermo2_CS, thermo2_SO);

void setup() {
  Serial.begin(9600);
  // Đợi module ổn định
  delay(500);
}

void loop() {
  // Khi C# gửi ký tự 'T' (Temperature), Arduino sẽ trả về dữ liệu
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'T') {
      float t1 = roomTemp.readCelsius();
      float t2 = coreTemp.readCelsius();

      // Gửi chuỗi định dạng: TempRoom,TempCore
      Serial.print(t1);
      Serial.print(",");
      Serial.println(t2);
    }
  }
  delay(200);
}
