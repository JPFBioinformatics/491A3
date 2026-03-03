import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def minimize_energy(y,beta,eta,h,max_passes=20):
    """
    Minimizes energy of MRF for a noisy image x
    """
    x = y.copy()
    x_start = x.copy()
    x_half = None

    rows,cols = x.shape
    energies = []
    snapshots = []
    energies.append(total_energy(x,y,beta,eta,h))

    # pass over image and remove noise
    for p in range(max_passes):
        flips = 0

        # randomize coordinates
        coords = [(r,c) for r in range(rows) for c in range(cols)]
        np.random.shuffle(coords)

        for r,c in coords:

            xi = x[r,c]

            # examine neighborhood
            nb_sum = 0
            if r > 0:
                nb_sum += x[r-1,c]
            if r < rows-1:
                nb_sum += x[r+1,c]
            if c > 0:
                nb_sum += x[r,c-1]
            if c < cols-1:
                nb_sum += x[r,c+1]

            # calculate energy change
            delta_e = 2 * xi * (beta * nb_sum + eta * y[r,c] - h)

            # filp if energy decreases
            if delta_e < 0 :
                x[r,c] = -xi
                flips += 1

        # update energies list every pass
        energies.append(total_energy(x,y,beta,eta,h))
        snapshots.append(x.copy())

        # stop if you converge to a value before max_passes
        if flips == 0:
            break
        
        x_half = snapshots[len(snapshots)//2]

    return x, x_start, x_half, energies

def load_bw_image(path):
    """
    loads black and white image at location path
    """
    img = Image.open(path).convert("L")

    arr = np.array(img)

    bw = np.where(arr < 128, 1, -1)

    return bw

def save_image(x,path):
    img = np.where(x==1, 0, 255).astype(np.uint8)
    Image.fromarray(img,mode="L").save(path)

def total_energy(x,y,beta,eta,h):
    """
    calculates the total energy of a black and white noisy image matrix
    """

    if x.shape != y.shape:
        raise ValueError("x and y are different shapes")
    
    # calculate bias term
    bias = h * np.sum(x)

    # calculate smoothness term
    right_paris = np.sum(x[:,:-1] * x[:,1:])
    down_paris = np.sum(x[:-1,:] * x[1:,:])
    smoothness = -beta * (right_paris +  down_paris)

    # calculate fidelity term
    fidelity = -eta * np.sum(x*y)

    return float(bias + smoothness + fidelity)

def add_noise(x, noise_prob=0.1):
    noisy = x.copy()
    flip = np.random.rand(*x.shape) < noise_prob
    noisy[flip] *= -1
    return noisy

def main():

    # parameters
    beta = 1
    eta = 2
    h = 0.5
    max_passes = 20

    img = "Grid.jpg"
    bw = load_bw_image(img)
    noisy = add_noise(bw,noise_prob=0.1)

    x_end,x_start,x_half,energies = minimize_energy(noisy, beta, eta, h, max_passes)
    print(len(energies))

    plt.figure()
    plt.plot(range(len(energies)), energies)
    plt.xlabel("Pass Number")
    plt.ylabel("Energy E(x,y)")
    plt.title("Energy vs Passes")
    plt.tight_layout()
    plt.savefig("Q1P3.png", dpi=300)
    plt.show()

    save_image(x_start, "denoise_start.png")
    save_image(x_half,  "denoise_half.png")
    save_image(x_end,   "denoise_final.png")

if __name__ == "__main__":
    main()