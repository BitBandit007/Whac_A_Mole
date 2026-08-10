import os
os.environ.setdefault("SDL_VIDEODRIVER","dummy"); os.environ.setdefault("SDL_AUDIODRIVER","dummy")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config
from graphics.raster import Canvas
from game.board import render_background
from game.entities import build_holes
from game import hud
holes = build_holes()
c = Canvas(config.WINDOW_WIDTH, config.WINDOW_HEIGHT, config.COL_BACKDROP_TOP)
render_background(c, holes)
hud.draw_hud(c, {"score":1234,"lives":4,"time_left":41.0,"combo":6,"multiplier":2,
                 "difficulty":"NORMAL","life_pulse":0.0,"effect_name":None,"effect_color":config.COL_TEXT})
plt.imsave("captures/_hud5.png", c.pixels[:110])
print("lives:", config.STARTING_LIVES, "| EASY up_time:", config.DIFFICULTIES["EASY"]["up_time"], "| sens:", config.MOUSE_SENSITIVITY)
