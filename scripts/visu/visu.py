import matplotlib.pyplot as plt

from ppdr.models import PPD, DAv2, DAv2Cleaned
from ppdr.utils.data_loader import HypersimLoader

loader = HypersimLoader("data/hypersim_test_set", 10)
entry = next(iter(loader))
bgr = entry[0]
plt.imsave("results/bgr.png", bgr)


ppd = PPD()
pred = ppd.predict(bgr)
plt.imsave("results/ppd.png", pred)

dav2 = DAv2()
pred = dav2.predict(bgr)
plt.imsave("results/dav2.png", pred)

dav2_cleaned = DAv2Cleaned(dav2)
pred = dav2_cleaned.predict(bgr)
plt.imsave("results/dav2_cleaned.png", pred)
