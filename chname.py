import os, re


find = re.compile(r"^[^.]*")
cwd = os.getcwd()
if "Hausdorff/Code/Models/MeshsegModels" in cwd:
    pass
elif "Hausdorff/Code" in cwd:
    os.chdir(cwd + "/Models/MeshsegModels")
elif "Hausdorff" in cwd:
    os.chdir(cwd + "/Code/Models/MeshsegModels")
else:
    print("[ERROR] Path couldn't be found...")
    exit(1)

files = os.listdir()
for i in range(len(files)):
    pass
    # print(i)
    model_num = int(re.search(find, files[i]).group(0))
    if model_num <= 20:
        os.rename(files[i], "Human" + str(model_num) + ".off")
    elif model_num > 20 and model_num <= 40:
        os.rename(files[i], "Cup" + str(model_num) + ".off")
    elif model_num > 40 and model_num <= 60:
        os.rename(files[i], "Glasses" + str(model_num) + ".off")
    elif model_num > 60 and model_num <= 80:
        os.rename(files[i], "Airplane" + str(model_num) + ".off")
    elif model_num > 80 and model_num <= 100:
        os.rename(files[i], "Ant" + str(model_num) + ".off")
    elif model_num > 100 and model_num <= 120:
        os.rename(files[i], "Chair" + str(model_num) + ".off")
    elif model_num > 120 and model_num <= 140:
        os.rename(files[i], "Octopus" + str(model_num) + ".off")
    elif model_num > 140 and model_num <= 160:
        os.rename(files[i], "Table" + str(model_num) + ".off")
    elif model_num > 160 and model_num <= 180:
        os.rename(files[i], "Teddy" + str(model_num) + ".off")
    elif model_num > 180 and model_num <= 200:
        os.rename(files[i], "Hand" + str(model_num) + ".off")
    elif model_num > 200 and model_num <= 220:
        os.rename(files[i], "Plier" + str(model_num) + ".off")
    elif model_num > 220 and model_num <= 240:
        os.rename(files[i], "Fish" + str(model_num) + ".off")
    elif model_num > 240 and model_num <= 260:
        os.rename(files[i], "Bird" + str(model_num) + ".off")
    elif model_num > 280 and model_num <= 300:
        os.rename(files[i], "Armadillo" + str(model_num) + ".off")
    elif model_num > 300 and model_num <= 320:
        os.rename(files[i], "Bust" + str(model_num) + ".off")
    elif model_num > 320 and model_num <= 340:
        os.rename(files[i], "Mech" + str(model_num) + ".off")
    elif model_num > 340 and model_num <= 360:
        os.rename(files[i], "Bearing" + str(model_num) + ".off")
    elif model_num > 360 and model_num <= 380:
        os.rename(files[i], "Vase" + str(model_num) + ".off")
    elif model_num > 380 and model_num <= 400:
        os.rename(files[i], "Fourleg" + str(model_num) + ".off")
