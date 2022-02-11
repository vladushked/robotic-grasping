from RAS_Com import RAS_Connect
import time

s = RAS_Connect('/dev/ttyTHS0')

slp = 0.55

cycle = 0


s.effectorMovement(-200, 400, 60, 0)
s.coordinateRequest()
time.sleep(slp)

cycle +=1
print("Cycle = ")
print(cycle)

