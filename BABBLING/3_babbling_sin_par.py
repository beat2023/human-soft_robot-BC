# Babbling data using 2 types of functions

from math import sin
from random import uniform, randint
import matplotlib.pyplot as plt
import numpy as np


def sine_wave(n_samples, amplitude, omega):
    # Function producing n_samples time points of a sine (range [0, n_samples-1])
    wave = []
    for t in range(n_samples):
        value = (80*amplitude*sin(omega*t) + 80)  # to have a value in (0, 160)
        wave.append(value)
    return wave


def parabola_wave(n_samples, a):
    # Function producing n_samples time points of a normalized exponential
    wave_aux = []
    wave = []
    for t in range(n_samples):
        value = -pow(t-(n_samples/2), 2)
        wave_aux.append(value)
        # Normalization to have a wave between 0 and 1
    normalization_max = max(wave_aux)
    normalization_min = min(wave_aux)
    for ind in range(n_samples):
        normalized_sample = (wave_aux[ind] - normalization_min) / (normalization_max - normalization_min)
        wave.append(160 * a * normalized_sample)
    return wave


def line(lower_lim, upper_lim):
    # Pass linearly from lower_lim to upper lim
    wave = []
    flag = 0
    t = 1
    m = 16
    if lower_lim > upper_lim:
        m = m * (-1)
    while not flag:
        value = m * t + lower_lim
        wave.append(value)
        t += 1
        if ((lower_lim <= upper_lim) & (value >= upper_lim)) | ((lower_lim >= upper_lim) & (value <= upper_lim)):
            flag = 1
            wave.pop()  # remove the last appended value
    return wave


if __name__ == '__main__':
    n_sine = 200
    n_par = 50
    sine_all = [[0], [0], [0], [0], [0], [0]]
    par_all = [[0], [0], [0], [0], [0], [0]]
    phase_par = [0, 1/3, 2/3]
    phase_sin = [0, 1 / 12, 1 / 6]

    for j in range(2):
        for i in range(12):
            # Setting the range of the random parameters and the fixed ones
            values = np.linspace(0.3, 1, 7)
            rand_i = randint(0, len(values)-1)
            ampl_sine = values[rand_i]
            omega_sine = uniform(1E-1, 0.3)

            # Calculating the i^th waveforms
            sine_aux = sine_wave(n_sine, ampl_sine, omega_sine)
            if abs(sine_aux[0] - sine_all[j*3][-1]) > 16:
                sine_all[j*3].extend(line(sine_all[j*3][-1], sine_aux[0]))
            sine_all[j*3].extend(sine_aux)

        for i in range(50):
            par_aux = []
            values = np.linspace(0.3, 1, 7)
            rand_i = randint(0, len(values)-1)
            a_par = values[rand_i]
            for rep in range(2):
                par_aux.extend(parabola_wave(int(n_par/2), a_par))
            if abs(par_aux[0] - par_all[j*3][-1]) > 16:
                par_all[j*3].extend(line(par_all[j*3][-1], par_aux[0]))
            par_all[j*3].extend(par_aux)

        for chamb in range(1, 3):
            samples_phase_par = int(n_par * phase_par[chamb])
            samples_phase_sin = int(n_sine * phase_sin[chamb])
            for sample in range(len(sine_all[j*3])):
                if sample in range(samples_phase_sin):
                    sine_all[chamb+j*3].append(0)
                else:
                    sine_all[chamb+j*3].append(sine_all[j*3][sample - samples_phase_sin])
            for sample in range(len(par_all[j*3])):
                if sample in range(samples_phase_par):
                    par_all[chamb+j*3].append(0)
                else:
                    par_all[chamb+j*3].append(par_all[j*3][sample - samples_phase_par])

    length_sines = []
    length_pars = []
    for c in range(6):
        if abs(sine_all[c][-1]) > 0:
            sine_all[c].extend(line(sine_all[c][-1], 0))
            length_sines.append(len(sine_all[c]))
        if abs(par_all[c][-1]) > 0:
            par_all[c].extend(line(par_all[c][-1], 0))
            length_pars.append(len(par_all[c]))
    for c in range(6):
        while len(sine_all[c]) < max(length_sines):
            sine_all[c].append(0)
        print(f'New len: {len(sine_all[c])}')
        while len(par_all[c]) < max(length_pars):
            par_all[c].append(0)
        print(f'New len: {len(par_all[c])}')

    # Plot
    fig, axs = plt.subplots(2, 1)
    plt.suptitle('Babbling: sinusoids')
    axs[0].plot(np.array(sine_all[0][:])/160, 'r', label='chamber 1')
    axs[0].plot(np.array(sine_all[1][:])/160, 'g', label='chamber 2')
    axs[0].plot(np.array(sine_all[2][:])/160, 'y', label='chamber 3')
    axs[0].legend()
    axs[1].plot(np.array(sine_all[3][:])/160, 'b', label='chamber 4')
    axs[1].plot(np.array(sine_all[4][:])/160, 'c', label='chamber 5')
    axs[1].plot(np.array(sine_all[5][:])/160, 'k', label='chamber 6')
    axs[1].legend()
    plt.xlabel('Sample number')
    plt.ylabel('Pressure value (Bars)')
    plt.show()

    fig1, axs = plt.subplots(2, 1)
    plt.suptitle('Babbling: parabolas')
    axs[0].plot(np.array(par_all[0][:])/160, 'r', label='chamber 1')
    axs[0].plot(np.array(par_all[1][:])/160, 'g', label='chamber 2')
    axs[0].plot(np.array(par_all[2][:])/160, 'y', label='chamber 3')
    axs[0].legend()
    axs[1].plot(np.array(par_all[3][:])/160, 'b', label='chamber 4')
    axs[1].plot(np.array(par_all[4][:])/160, 'c', label='chamber 5')
    axs[1].plot(np.array(par_all[5][:])/160, 'k', label='chamber 6')
    axs[1].legend()
    plt.xlabel('Sample number')
    plt.ylabel('Pressure value (Bars)')
    plt.show()

    # Plot of differences
    sine_diff = [[], [], [], [], [], []]
    for i in range(len(sine_all)):
        for j in range(len(sine_all[i])-1):
            diff = sine_all[i][j+1] - sine_all[i][j]
            sine_diff[i].append(diff)
    par_diff = [[], [], [], [], [], []]
    for i in range(len(par_all)):
        for j in range(len(par_all[i]) - 1):
            diff = par_all[i][j + 1] - par_all[i][j]
            par_diff[i].append(diff)

    fig_diff, axs = plt.subplots(4, 1)
    plt.suptitle("Difference in pressures")
    axs[0].plot(sine_diff[0][:], 'r')
    axs[0].plot(sine_diff[1][:], 'g')
    axs[0].plot(sine_diff[2][:], 'b')

    axs[1].plot(sine_diff[3][:], 'r')
    axs[1].plot(sine_diff[4][:], 'g')
    axs[1].plot(sine_diff[5][:], 'b')

    axs[2].plot(par_diff[0][:], 'r')
    axs[2].plot(par_diff[1][:], 'g')
    axs[2].plot(par_diff[2][:], 'b')

    axs[3].plot(par_diff[3][:], 'r')
    axs[3].plot(par_diff[4][:], 'g')
    axs[3].plot(par_diff[5][:], 'b')
    plt.show()

    ####################################################################################################################
    # SAVE DATA
    # rev = 0
    # file_wname = 'DataFolder/sinusoids_' + str(rev) + '.txt'
    # with open(file_wname, 'w') as f:
    #     for d in range(len(sine_all[0])):
    #         f.write(str([d, sine_all[0][d], sine_all[1][d], sine_all[2][d],
    #                      sine_all[3][d], (sine_all[4][d]/160)*52, sine_all[5][d]]).strip('[]'))
    #         f.write('\n')
    #
    # file_wname2 = 'DataFolder/parabola_' + str(rev) + '.txt'
    # with open(file_wname2, 'w') as f2:
    #     for d in range(len(par_all[0])):
    #         f2.write(str([d, par_all[0][d], par_all[1][d], par_all[2][d],
    #                       par_all[3][d], (par_all[4][d]/160)*52, par_all[5][d]]).strip('[]'))
    #         f2.write('\n')
