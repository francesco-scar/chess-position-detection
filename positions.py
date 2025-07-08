from os import listdir
from os.path import isfile
from json import load as jsonLoad
import numpy as np
import math

import matplotlib.pyplot as plt
from PIL import Image

fig, axs = plt.subplots(2)

def calc_position(box, corners, side):

    left = sorted(corners, key=lambda corner: corner[0], reverse=False)[0:2]
    right = sorted(corners, key=lambda corner: corner[0], reverse=True)[0:2]
    upper = sorted(corners, key=lambda corner: corner[1], reverse=True)[0:2]
    lower = sorted(corners, key=lambda corner: corner[1], reverse=False)[0:2]

    ll = {"x":corners[0][0], "y":corners[0][1]}
    lr = {"x":corners[0][0], "y":corners[0][1]}
    ul = {"x":corners[0][0], "y":corners[0][1]}
    ur = {"x":corners[0][0], "y":corners[0][1]}

    for i in range(1, 4):
        if corners[i] in left and corners[i] in lower:
            ll = {"x":corners[i][0], "y":corners[i][1]}
            continue
        if corners[i] in right and corners[i] in lower:
            lr = {"x":corners[i][0], "y":corners[i][1]}
            continue
        if corners[i] in left and corners[i] in upper:
            ul = {"x":corners[i][0], "y":corners[i][1]}
            continue
        if corners[i] in right and corners[i] in upper:
            ur = {"x":corners[i][0], "y":corners[i][1]}
            continue

    proj_pos = {
        "x": box[0] + box[2] / 2,
        "y": box[1] + box[3] * 0.95
    }

    axs[0].scatter(proj_pos["x"], proj_pos["y"], s=10)

    # https://math.stackexchange.com/questions/2892004/calculate-a-location-on-a-square-on-squares-projection-on-a-plane
    # https://math.stackexchange.com/questions/296794/finding-the-transform-matrix-from-4-projected-points-with-javascript/339033#339033

    A = np.array([  [ll["x"], lr["x"], ul["x"]], 
                    [ll["y"], lr["y"], ul["y"]], 
                    [1,       1,       1      ]])
    
    b = np.array(   [ur["x"], ur["y"], 1      ])

    x = np.linalg.solve(A, b)

    A_scaled = np.array([   [ll["x"] * x[0], lr["x"] * x[1], ul["x"] * x[2]], 
                            [ll["y"] * x[0], lr["y"] * x[1], ul["y"] * x[2]], 
                            [x[0],           x[1],           x[2]          ]])
    
    invA = np.linalg.inv( A_scaled )

    B_scaled = np.array([   [ 0, 1, 0], 
                            [ 0, 0, 1], 
                            [-1, 1, 1]])
    
    C = np.dot(B_scaled, invA)

    proj_pos_vec = np.array([proj_pos["x"], proj_pos["y"], 1])

    res_pos_vec = np.dot(C, proj_pos_vec)

    pos = {
        "x": res_pos_vec[0]/res_pos_vec[2],
        "y": 1-res_pos_vec[1]/res_pos_vec[2] # (inversione: nelle immagini più alto vuol dire più vicino all'osservatore)
        #"y": res_pos_vec[1]/res_pos_vec[2]
    }

    square_pos = {
        "x": math.floor(pos["x"]*8),
        "y": math.floor(pos["y"]*8)
    }

    axs[1].scatter(pos["x"], pos["y"], s=10)

    if square_pos["x"] < 0 or square_pos["x"] > 7 or square_pos["y"] < 0 or square_pos["y"] > 7:
        return "ERRORE: fuori dalla scacchiera!"

    letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    match side:
        case 0:
            # 01 11 -> A2 B2
            # 00 10 -> A1 B1
            return letters[square_pos["x"]]+str(square_pos["y"]+1)
        case 1:
            # 01 11 -> B2 B1
            # 00 10 -> A2 A1
            return letters[square_pos["y"]]+str(8-square_pos["x"])
        case 2:
            # 01 11 -> B1 A1
            # 00 10 -> B2 A2
            return letters[7-square_pos["x"]]+str(8-square_pos["y"])
        case 3:
            # 01 11 -> A1 A2
            # 00 10 -> B1 B2
            return letters[7-square_pos["y"]]+str(square_pos["x"]+1)

dir = "./dataset/train/"

total = 0
ok = 0
for file in listdir(path=dir):
    if isfile(dir+file) and file.endswith(".json"):
        with open(dir+file) as f:
            content = jsonLoad(f)
            side = 0 if content["white_turn"] else 2

            img = Image.open(dir+file.replace(".json", ".png"))
            axs[0].imshow(img)
            x, y = zip(*content["corners"])
            axs[0].plot(x + (x[0],), y + (y[0],), color='cyan', linewidth=1)  # connect corners in order and close the box

            for i in range(1,8):
                axs[1].plot([i/8, i/8], [0, 1], color='red', linewidth=0.5)
                axs[1].plot([0, 1], [i/8, i/8], color='red', linewidth=0.5)
            axs[1].plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], color='cyan', linewidth=1)

            for piece in content["pieces"]:
                rect = plt.Rectangle((piece["box"][0], piece["box"][1]), piece["box"][2], piece["box"][3],
                                    linewidth=1, edgecolor='lime', facecolor='none')
                axs[0].add_patch(rect)

                position = calc_position(piece["box"], content["corners"], side)
                
                total += 1
                if position == piece["square"]:
                    ok += 1
            
            axs[1].set_aspect('equal', adjustable='box')
            #plt.show()
            fig, axs = plt.subplots(2)


print(f"{ok}/{total}: {round(100*ok/total, 2)}%")
            