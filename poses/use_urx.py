import urx
from urx import Robot
ur5 = urx.Robot('192.168.0.103')
homej = [0.0001, -1.1454, -2.7596, 0.7290, 0.0000, 0.0000]
startGj = [0.2065, -1.7696, -0.4635, -2.3716, 1.5644, 0.2056]
ur5.movej(home1j, 0.4, 0.4)
ur5.movej(startGj, 0.4, 0.4)
ur5.stop()
ur5.close()
exit()
