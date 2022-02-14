from RAS_Com import RAS_Connect
import time

s = RAS_Connect('/dev/ttyTHS0')

slp = 0.55

cycle = 0


s.effectorMovement(0, 200, 500, 0)
s.coordinateRequest()
time.sleep(slp)

cycle +=1
print("Cycle = ")
print(cycle)

