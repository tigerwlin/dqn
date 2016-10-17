import numpy as np
import matplotlib.pyplot as plt



# fileList = ['/tmp/tmp26EjTG/record.log', '/tmp/tmpFRc4B5/record.log', '/tmp/tmpXtuXR_/record.log', '/tmp/tmpQ7QdEl/record.log']
# fileList = ['/mnt/scratch/wulin/tmp/tmpNCZRO3_TimePilot-v0/record.log', '/mnt/scratch/wulin/tmp/tmptyieEO_TimePilot-v0/record.log']
# fileList = ['/mnt/scratch/wulin/tmp/tmpQO10E0_Zaxxon-v0/record.log', '/mnt/scratch/wulin/tmp/tmpFL34J2_Zaxxon-v0/record.log']
# fileList = ['/mnt/scratch/wulin/tmp/tmpEZ7ryr_Alien-v0/record.log', '/mnt/scratch/wulin/tmp/tmpiXpd75_Alien-v0/record.log']
fileList = ['/mnt/scratch/wulin/tmp/tmpp6Y14O/record.log', '/mnt/scratch/wulin/tmp/tmp16gX6W/record.log']   # SpaceInvader?

num_plots = len(fileList)
plot_id = 0
average_steps = 125000
# _, axs = plt.subplots(1, num_plots, sharey=True)

for fname in fileList:
    dqn_value = []
    is_doubleDQN = False
    with open(fname, 'r') as f:
        for line in f:
            # if "doubleDQN" in line:
            #     is_doubleDQN = True
            #     continue
            line_str = line.split()
            dqn_value.append(float(line_str[-1]))

    tail_cutoff = len(dqn_value) % average_steps
    if tail_cutoff != 0:
        dqn_value = np.array(dqn_value[:-tail_cutoff])
    else:
        dqn_value = np.array(dqn_value)
    dqn_value = dqn_value.reshape((-1, average_steps))
    dqn_value = np.mean(dqn_value, axis=1)
    if "doubleDQN" in line:
        is_doubleDQN = True

    # =================================================
    # axs[plot_id].plot(dqn_value)
    # if is_doubleDQN:
    #     axs[plot_id].set_title("doubleDQN value")
    # else:
    #     axs[plot_id].set_title("DQN value")
    # plot_id += 1

    x = (np.arange(dqn_value.shape[0])+1)*125000/1e6
    if is_doubleDQN:
        plt.plot(x, dqn_value,'b')
    else:
        plt.plot(x, dqn_value, 'r')
    plt.title("value estimates by doubleDQN (blue) & DQN (red)")
    plt.xlabel("steps in millions")
plt.show()

