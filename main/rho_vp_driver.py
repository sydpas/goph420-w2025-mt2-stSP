import numpy as np
import matplotlib.pyplot as plt
from question2.regression import (multi_regress)


def main():
    data = np.loadtxt("../data/Question_2_DATA_rho_vp.txt")

    rho = data[:, 0]  # density data
    vp = data[:, 1]  # p-wave velocity data

    # plotting original data...
    plt.grid()
    plt.plot(rho, vp, 'ro', markersize=4)
    plt.xlabel("Density [g/cm^3]")
    plt.ylabel("P-Wave Velocity [m/s]")
    plt.title("P-Wave Velocity as a Function of Density")
    plt.savefig('../figures/magnitude_vs_time.png', dpi=300)
    plt.show()

    # linearizing (2b)...
    y = np.log10(vp)  # linearize the relationship
    Z = np.vstack((np.ones_like(rho), rho)).T  # design matrix for regression

    a, e, rsq = multi_regress(y, Z)

    log_n = Z @ a  # linearized vp values

    # plotting...
    plt.grid()
    plt.plot(rho, log_n, 'b--', markersize=2, label="Fitted")  # density-x vs log(vp)-y
    plt.plot(rho, y, 'ro', markersize=4, label="Log Data")
    plt.xlabel("Density [g/cm^3]")
    plt.ylabel("Log of P-Wave Velocity [m/s]")
    plt.legend()
    plt.title("Linear Regression: log(Vp) versus Density")
    plt.savefig('../figures/linear_magnitude_vs_time.png', dpi=300)
    plt.show()

    # now we fit the model in OG
    log_V0, k = a
    V0 = 10 ** log_V0  # undo the log to get actual V0
    vp_fitted = V0 * rho ** k  # equation from midterm

    print(f"V0 = {V0}")
    print(f"k = {k}")
    print(f"R squared = {rsq}")

    # plotting original data with model fit...
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(rho, vp_fitted, 'b--', markersize=2, label="Fitted Line")  # density-x vs log(vp)-y
    plt.plot(rho, vp, 'ro', markersize=4, label="Original Data")
    plt.xlabel("Density [g/cm^3]")
    plt.ylabel("P-Wave Velocity [m/s]")
    plt.legend()
    plt.title("Best Fit Model")
    plt.savefig('../figures/best_fit.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
