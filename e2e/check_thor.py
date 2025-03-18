# Code from https://github.com/askforalfred/alfred/blob/master/scripts/check_thor.py

from ai2thor.controller import Controller

c = Controller()
c.start()
event = c.step(dict(action="MoveAhead"))
assert event.frame.shape == (300, 300, 3)
print(event.frame.shape)
print("Everything works!!!")