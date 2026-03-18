import spectral as spy
import numpy as np
import matplotlib.pyplot as plt

# Path to your SPECIM image (the .hdr file)
#hdr_path = "/Users/cindygrimm/VSCode/data/cherry/2025-12-12_001/"
hdr_path = "/Users/cindygrimm/VSCode/data/cherry/2025-12-12_001/results/REFLECTANCE_2025-12-12_001.hdr"

# Load the hyperspectral image
img = spy.open_image(hdr_path)

# Convert to a numpy array (optional, but useful)
data = img.load()

print("Image shape (rows, cols, bands):", data.shape)

# Display a quick RGB composite
# SPECIM cameras often use bands around:
# R ≈ 60, G ≈ 30, B ≈ 10 (this varies by model!)
rgb = spy.get_rgb(img, [60, 30, 10])

plt.figure(figsize=(8, 6))
plt.imshow(rgb)
plt.title("SPECIM RGB Composite")
plt.axis("off")
plt.show()
