import numpy as np
import matplotlib.pyplot as plt
import tyro
import cv2



def f(x, y):
    return np.sin(x*y) * np.exp(-(x**2 + y**2) / 10) 

def make_population(n, dom):
    return np.random.uniform(-dom, dom, (n, 2))

def score(population):
    return f(population[:, 0], population[:, 1])


def get_viridis_heatmap(data):
    # Apply the viridis colormap
    colormap = plt.cm.viridis
    normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    mapped_data = colormap(normed_data)

    # to uint8
    mapped_data = (255 * mapped_data).astype(np.uint8)

    # Return the RGBA values
    return mapped_data




def main(rounds: int = 100, mutation_rate: float = 0.1, pixels: int = 1000, pop_size: int = 100):

    dom = 5
    x = np.linspace(-dom, dom, pixels)
    y = np.linspace(-dom, dom, pixels)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # make population
    population = make_population(pop_size, dom)

    populations = [population]

    curr_population = population

    for round in range(rounds):
        scores = score(curr_population)
        if round % 10 == 0:
            print(f'Round {round+1} - Best score: {scores.max():.2f}  Worst score: {scores.min():.2f}  Mean score: {scores.mean():.2f}')
        new_population = curr_population[scores.argsort()[-pop_size:]]  # Only the top 50% survive
        new_population = np.repeat(new_population, 2, axis=0)  # Each individual has 2 children
        new_population += np.random.uniform(-mutation_rate, mutation_rate, new_population.shape)  # Mutate the children a bit

        populations.append(new_population)
        curr_population = new_population

    heatmap = get_viridis_heatmap(Z)[:, :, :3]

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    for population in populations:
        frame = heatmap.copy()
        print(frame.shape, type(frame), frame.dtype)

        for x, y in population:
            scale = (pixels / 2) / dom
            shift = pixels / 2

            center = (int(x * scale + shift), int(y * scale + shift))

            cv2.circle(
                frame, 
                center,
                10, 
                (0, 0, 255),
                cv2.FILLED,
                8,
                0
            )

        cv2.imshow('frame', frame)

        key = cv2.waitKey(0)
        char_key = chr(key & 0xFF)

        if char_key == 'q':
            break


if __name__ == "__main__":
    tyro.cli(main)