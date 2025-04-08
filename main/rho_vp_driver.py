import numpy as np
import matplotlib.pyplot as plt
from question2.regression import (multi_regress)


def main():
    data = np.loadtxt("../data/Question_2_DATA_rho_vp.txt")

    rho = data[:, 0]  # density data
    vp = data[:, 1]  # p-wave velocity data

    # plotting...
    plt.grid()
    plt.plot(rho, vp, 'ro', markersize=4)
    plt.xlabel("Density [g/cm^3]")
    plt.ylabel("P-Wave Velocity [m/s]")
    plt.title("P-Wave Velocity as a Function of Density")
    plt.savefig('../figures/magnitude_vs_time.png', dpi=300)
    plt.show()


    # linearizing (2b)
    m = rho
    n = vp  # array of zeros same size as m

    y = np.log10(n)  # linearize the relationship
    Z = np.vstack((np.ones_like(m), m)).T  # design matrix for regression

    a, e, rsq = multi_regress(y, Z)  # use least square regression to find coeffs, residuals, and R^2

    log_n = Z @ a  # linearized vp values

    # plotting...
    plt.grid()
    plt.plot(m, log_n, 'bo', markersize=2)  # density vs log
    plt.xlabel("Density [g/cm^3]")
    plt.ylabel("P-Wave Velocity [m/s]")
    plt.title("Linear Regression: P-Wave Velocity as a Function of Density")
    plt.savefig('../figures/linear_magnitude_vs_time.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()