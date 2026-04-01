import matplotlib.pyplot as plt
import torch

from ppdr.models import PPD, DAv2, DAv2Cleaned
from ppdr.utils.reader import HypersimReader

loader = HypersimReader("data/hypersim_test_set")
entry_name, rgb, distances, ndc_to_cam = next(iter(loader))
plt.imsave("results/rgb.png", rgb)

rgb_tensor = torch.from_numpy(rgb)


ppd = PPD()
pred = ppd.predict(rgb_tensor).cpu().numpy()
plt.imsave("results/ppd.png", pred)

dav2 = DAv2()
pred = dav2.predict(rgb_tensor).cpu().numpy()
plt.imsave("results/dav2.png", pred)

dav2_cleaned = DAv2Cleaned(dav2)
pred = dav2_cleaned.predict(rgb_tensor).cpu().numpy()
plt.imsave("results/dav2_cleaned.png", pred)
