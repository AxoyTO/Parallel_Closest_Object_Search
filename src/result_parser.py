import os
import re
import pandas as pd


def go_to_dir():
    if "Hausdorff" in os.getcwd():
        while True:
            dirs = os.listdir(os.getcwd())
            if "Results" in dirs:
                break
            os.chdir("..")
        os.chdir("Results")


def parse_data():
    procs = []
    times = []
    speedups = []
    with open("result.out", "r") as file:
        for line in file:
            if "WORLD SIZE:" in line:
                procs.append((int(re.search("\d+", line).group(0))))
            if "Parallel Elapsed Time:" in line:
                time = float(re.search("(\d+.\d+).", line).group(0))
                times.append(time)
                speedups.append(times[0] / time)

    procs.remove(4)
    df = pd.DataFrame({"PROCESSES": procs, "TIMES(s)": times, "SPEEDUP": speedups})
    return df


def write_to_excel(df):
    with pd.ExcelWriter("results.xlsx") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
    pass


if __name__ == "__main__":
    go_to_dir()
    data = parse_data()
    write_to_excel(data)
