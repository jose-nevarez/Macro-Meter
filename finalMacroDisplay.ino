#include <LiquidCrystal.h>


LiquidCrystal lcd(12,11,5,4,3,2); 
void setup() {
  lcd.begin(16, 2);
  lcd.setCursor(0,0); //select the row to print
  lcd.print("Macros | Cal: ");
  lcd.setCursor(0,1); 
  lcd.print("Fat:|Prot:");
}

void loop() {
}
