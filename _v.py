import os
os.environ.setdefault("SDL_VIDEODRIVER","dummy"); os.environ.setdefault("SDL_AUDIODRIVER","dummy")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config
from graphics.raster import Canvas
from game.board import render_background
from game.entities import build_holes
from game import hud, highscore

for s,m,h,a in [(4820,"NORMAL",52,82.5),(3110,"HARD",41,76.0),(6650,"EASY",71,90.2),(1,"HARD",1,2.0)]:
    highscore.record(s,m,h,a)

holes = build_holes()
bg = Canvas(config.WINDOW_WIDTH, config.WINDOW_HEIGHT, config.COL_BACKDROP_TOP)
render_background(bg, holes)
for mode in ("EASY","HARD"):
    c = bg.clone(); hud.draw_menu(c, mode, highscore.best_score(mode), 1.0)
    plt.imsave(f"captures/_menu_{mode}.png", c.pixels)
c = bg.clone()
hud.draw_game_over(c, {"reason":"time","score":4200,"hits":48,"misses":9,"escaped":3,
    "accuracy":84.2,"best_combo":4,"difficulty":"NORMAL","is_new_best":False,"mode_best":4820}, 1.0)
plt.imsave("captures/_gameover.png", c.pixels)
print("EASY best:", highscore.best_score("EASY"), "NORMAL:", highscore.best_score("NORMAL"), "HARD:", highscore.best_score("HARD"))
