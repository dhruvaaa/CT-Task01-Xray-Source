import numpy as np
import matplotlib.pyplot as plt


kVp = 120    
mA = 1200        
Z = 74            
filter_thickness = 1.5 


energies = np.arange(1.0, kVp + 1.0, 0.5)


unfiltered = mA * Z * (kVp - energies) / energies
unfiltered = np.maximum(unfiltered, 0)


transmission = np.exp(-filter_thickness * (25.0 / energies) ** 3)
spectrum = unfiltered * transmission


plt.figure(figsize=(9, 4.5))
plt.plot(energies, spectrum, color="navy", lw=2)
plt.title(f"X-Ray Emission Spectrum ({kVp} kVp, W Target)")
plt.xlabel("Energy (keV)")
plt.ylabel("Relative Photon Intensity")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlim(0, kVp + 5)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()
