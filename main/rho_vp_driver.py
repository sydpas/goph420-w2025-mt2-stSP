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
    plt.title("Original Data", fontsize=16, fontweight='bold')
    plt.savefig('../figures/og_plot.png', dpi=300)
    plt.show()

    # linearizing (2b)...
    y = np.log(vp)  # linearize the relationship
    Z = np.vstack((np.ones_like(rho), rho)).T  # design matrix for regression

    a, e, rsq = multi_regress(y, Z)

    log_n = Z @ a  # linearized vp values

    # plotting log-linear fit...
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(rho, log_n, 'b--', markersize=2, label="Fitted Line")  # density-x vs log(vp)-y
    plt.plot(rho, y, 'ro', markersize=6, alpha=0.6, label="Log Data")
    plt.xlabel("Density [g/cm³]", fontsize=12)
    plt.ylabel("log(P-Wave Velocity [m/s])", fontsize=12)
    plt.title('Linearized Data With Regression Line', fontsize=16, fontweight='bold')
    plt.legend()
    plt.savefig('../figures/linear_with_regress.png', dpi=300)
    plt.show()


    # now we fit the model
    k = a[1]
    v0 = np.exp(a[0])  # undo the log to get actual V0
    vp_new = v0 * np.exp(k * rho)

    index = np.argsort(rho)
    rho_sorted = rho[index]
    vp_sorted = vp_new[index]

    # plotting original data with exponential...
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(rho_sorted, vp_sorted, 'b--', markersize=2, label="Fitted Line")
    plt.plot(rho, vp,'ro', markersize=6, alpha=0.6, label="Original Data")
    plt.text(0.20, 0.60, f'$R^2$ = {rsq:.8f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white'))
    plt.text(0.20, 0.70, rf'$V_p = {v0:.2f} \cdot e^{{\rho \times {k:.2f}}}$', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white'))
    # transform is to place text relative to the axes, not data
    plt.xlabel("Density [g/cm³]", fontsize=12)
    plt.ylabel("log(P-Wave Velocity [m/s])", fontsize=12)
    plt.title("Original Data with Regression Line", fontsize=16, fontweight='bold')
    plt.legend()
    plt.savefig('../figures/og_with_exp.png', dpi=300)
    plt.show()

    print(f"V0 = {v0}")
    print(f"k = {k}")
    print(f"R squared = {rsq}")


if __name__ == "__main__":
    main()
