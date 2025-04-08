import numpy as np
import matplotlib.pyplot as plt
from question2.regression import (multi_regress)

# raw, log(raw) with straight, raw with curved


def main():
    data = np.loadtxt("../data/Question_2_DATA_rho_vp.txt")

    rho = data[:, 0]  # density data
    vp = data[:, 1]  # p-wave velocity data

    # plotting original data...
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(rho, vp, 'ro', markersize=6, alpha=0.6, label="Original Data")
    plt.xlabel("Density [g/cm³]", fontsize=12)
    plt.ylabel("P-Wave Velocity [m/s]", fontsize=12)
    plt.title("P-Wave Velocity vs Density", fontsize=14)
    plt.legend()
    plt.savefig('../figures/magnitude_vs_time.png', dpi=300)
    plt.show()

    # linearizing (2b)...
    y = np.log(vp)  # linearize the relationship
    Z = np.vstack((np.ones_like(rho), rho)).T  # design matrix for regression

    a, e, rsq = multi_regress(y, Z)

    log_n = Z @ a  # linearized vp values

    # now we fit the model
    k = a[1]
    v0 = np.exp(a[0])  # undo the log to get actual V0

    print(f"V0 = {v0}")
    print(f"k = {k}")
    print(f"R squared = {rsq}")

    # plotting log-linear fit...
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(rho, log_n, 'b--', markersize=2, label="Fitted Line")  # density-x vs log(vp)-y
    plt.plot(rho, y, 'ro', markersize=6, alpha=0.6, label="Log Data")
    plt.xlabel("Density [g/cm³]", fontsize=12)
    plt.ylabel("log(P-Wave Velocity [m/s])", fontsize=12)
    plt.legend()
    plt.savefig('../figures/best_fit.png', dpi=300)
    plt.show()

    # plotting original data with exponential...
    plt.title(rf"Linear Regression of $V_p = {v0:.2f} \cdot e^{{\rho \times {k:.2f}}}$", fontsize=14)
    plt.text(0.20, 0.60, f'$R^2$ = {rsq:.8f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white'))
    # transform is to place text relative to the axes, not data




if __name__ == "__main__":
    main()
