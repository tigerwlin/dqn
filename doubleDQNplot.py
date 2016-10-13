import numpy as np
import matplotlib.pyplot as plt



fileList = ['/tmp/tmp26EjTG/record.log', '/tmp/tmpFRc4B5/record.log', '/tmp/tmpXtuXR_/record.log', '/tmp/tmpQ7QdEl/record.log']
num_plots = len(fileList)
plot_id = 0
_, axs = plt.subplots(1, num_plots, sharey=True)

for fname in fileList:
    dqn_value = []
    is_doubleDQN = False
    with open(fname, 'r') as f:
        for line in f:
            line_str = line.split()
            dqn_value.append(float(line_str[-1]))

    if "doubleDQN" in line:
        is_doubleDQN = True

    # plt.subplot((12, plot_id))
    axs[plot_id].plot(dqn_value)
    if is_doubleDQN:
        axs[plot_id].set_title("doubleDQN value")
    else:
        axs[plot_id].set_title("DQN value")
    plot_id += 1

plt.show()

