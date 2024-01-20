# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: project-pVskOJpn-py3.10
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import mediapy


# %%

def f(x, y):
    return np.sin(x*y) * np.exp(-(x**2 + y**2) / 10) 


# %%
pixels = 100

dom = 5
x = np.linspace(-dom, dom, pixels)
y = np.linspace(-dom, dom, pixels)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 3D Plotting
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.plot_surface(X, Y, Z, alpha=0.7)

# ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, rstride=10, cstride=10)  # Adjust alpha for transparency


ax.set_title('3D Polynomial Plot')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()


# %%

def heatmap_plot(population):
    # Plotting a heatmap and scattering points over it
    fig, ax = plt.subplots(figsize=(8, 6))

    # Heatmap plot
    # Note: We're using the Z values calculated from the polynomial function.
    # 'aspect' argument to 'auto' so that the plot isn't squashed

    im = ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis', aspect='auto')

    # Adding a colorbar
    fig.colorbar(im, ax=ax)

    # Scatter plot - overlaying on the heatmap
    # We'll use the same random points as before for the scatter plot
    if population is not None:
        scatter = ax.scatter(population[:, 0], population[:, 1], color='red', s=50)  # 's' is the size of the points

    ax.set_title('Heatmap with Scatter Points')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    plt.show()



# %%
# make population
def make_population(n):
    return np.random.uniform(-dom, dom, (n, 2))


# %%
population = make_population(100)

# %%
heatmap_plot(population)


# %%
def score(population):
    return f(population[:, 0], population[:, 1])


# %%
populations = [population]

curr_population = population
rounds = 100
mutation_rate = 0.1

for round in range(rounds):
    scores = score(curr_population)
    if round % 10 == 0:
        print(f'Round {round+1} - Best score: {scores.max():.2f}  Worst score: {scores.min():.2f}  Mean score: {scores.mean():.2f}')
    new_population = curr_population[scores.argsort()[-50:]]  # Only the top 50% survive
    new_population = np.repeat(new_population, 2, axis=0)  # Each individual has 2 children
    new_population += np.random.uniform(-mutation_rate, mutation_rate, new_population.shape)  # Mutate the children a bit

    populations.append(new_population)
    curr_population = new_population

# %%
len(populations)


# %%
def get_viridis_heatmap(data):
    # Apply the viridis colormap
    colormap = plt.cm.viridis
    normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    mapped_data = colormap(normed_data)

    # to uint8
    mapped_data = (255 * mapped_data).astype(np.uint8)

    # Return the RGBA values
    return mapped_data


# %%
heatmap = get_viridis_heatmap(Z)[:, :, :3]
display(heatmap.shape, heatmap.dtype)
mediapy.show_image(heatmap)

# %%
import cv2
vid = []

window = cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

for population in populations:
    frame = heatmap.copy()
    print(frame.shape, type(frame), frame.dtype)

    for x, y in population:
        cv2.circle(
            frame, 
            (
                int(x*(dom * 2)+pixels/2), 
                int(y*(dom * 2)+pixels/2)
            ), 
            1, 
            (0, 0, 255),
            1,
            8,
            0
        )

    cv2.imshow('frame', frame)


    key = cv2.waitKey(0)

    print(frame.shape)

    vid.append(score(population).max())
