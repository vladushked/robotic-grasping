import serial
import time
import keyboard
#from typing import Sequence
import random

class RAS_Connect():
    def __init__(self, _port):
        ser = serial.Serial (port=_port,
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False)
        self.ser = ser
        

    def __convertCommand(self, str1):
        st = str1 + " " + chr(random.randrange(97, 97 + 26)) + chr(random.randrange(97, 97 + 26))

        st = bytearray(st, encoding = 'utf-8')
        #st[4] = len(st) + 2
        cs = st[0]
        for j in st[1:]:
            cs ^= j
        st += bytearray("*", encoding = 'utf-8')
        st += bytearray(chr(cs), encoding = 'utf-8')
        st += bytearray(str.encode('\r'))
        print(st)
        return st
    
    def __con(self, s):
        st = ""
        if s >= 0:
            st = "+"
        else:
            st = "-"
        s = abs(s)
        if s >= 100:
            st += str(int(s))[:3]
            return st
        if s >= 10 and s < 100:
            st += "0" + str(int(s))[:2]
            return st
        if s < 10:
            st += "00" + str(int(s))[:1]
            return st
    
    def __conWS(self, s):
        st = ""
        s = abs(s)
        if s >= 100:
            st += str(int(s))[:3]
            return st
        if s >= 10 and s < 100:
            st += "0" + str(int(s))[:2]
            return st
        if s < 10:
            st += "00" + str(int(s))[:1]
            return st

    def __sendMes(self, st):
        self.ser.write(st)
        received_data = bytearray()
        start_time = time.time()
        while (time.time() - start_time < 0.1):
            while self.ser.inWaiting() > 0:
                received_data += self.ser.read(1)
        print (received_data)
        return received_data

    def rodRotation(self, A, B, C, D, E):
        st = self.__sendMes(self.__convertCommand("G00 A" + self.__con(A) + " B" + self.__con(B) + " C" + self.__con(C) + " D" + self.__con(D) + " E" + self.__con(E)))
        try:
            return st[4]
        except:
            return ""

    def effectorMovement(self, X, Y, Z, R):
        st = self.__sendMes(self.__convertCommand("G01 X" + self.__con(X) + " Y" + self.__con(Y) + " Z" + self.__con(Z) + " R" + self.__con(R)))
        try:
            return st[4]
        except:
            return ""
    
    def home(self):
        st = self.__sendMes(self.__convertCommand("G28"))
        try:
            return st[4]
        except:
            return ""

    def transportation(self):
        st = self.__sendMes(self.__convertCommand("G30"))
        try:
            return st[4]
        except:
            return ""
    
    def angleRequest(self):
        st = self.__sendMes(self.__convertCommand("G25"))
        return st

    def coordinateRequest(self):
        st = self.__sendMes(self.__convertCommand("G26"))
        return st

    def grip(self, A):
        st = self.__sendMes(self.__convertCommand("M03 " + self.__conWS(A)))
        try:
            return st[4]
        except:
            return ""

    def expansion(self):
        self.__sendMes(self.__convertCommand("M05"))
    
    def expWithLifting(self):
        self.__sendMes(self.__convertCommand("M06"))


#s = ComRAS('/dev/ttyTHS0')

#s.rodRotation(100, -100, 25, 9, 7)
#s.effectorMovement(123, 150, 0, 5)
#s.home()
#s.transportation()
#s.angleRequest()
#s.coordinateRequest()
#s.grip(30)
#s.expansion()
#s.expWithLifting()
#s.home()
