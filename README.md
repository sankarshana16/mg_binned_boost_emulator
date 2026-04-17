# MG Boost Emulator

A fast emulator for modified gravity (MG) boosts to the matter power spectrum.

This emulator combines:
- A neural network (NN) for linear scales
- A Gaussian Process (GP) emulator for nonlinear scales
- A smooth stitching procedure across k

It supports redshift-dependent modified gravity models with binning.

---

##  Features

- Accurate to ~1% across parameter space
- Supports multiple redshifts
- Handles MG parameters (μ, η) and bin-dependent activation
- Fully replaces expensive Boltzmann / simulation calls in inference pipelines

---

## Repository Structure
```
mg_boost_emulator/
│
├── emulator/ # Core emulator code
├── src/ # Required for loading pickled Standardizer (bin 5 GP)
├── models/ # Pretrained models (not included)
├── examples/ # Demo script
│
├── README.md
├── requirements.txt

```
---

## Model Files

Pretrained models are **not included** in this repository due to size.

Download them from:

👉 **[INSERT LINK HERE]**

and place them in models/

---

Required files:
```
linear_boost_nn.pt
gp_full_corrected.cpk
gp_bin5.cpk
pca_full_corrected.cpk
pca_bin5.cpk
standardizer_bin5.cpk
cola_eg.txt
```
---

## Installation

Clone the repo:

```python
git clone https://github.com/yourusername/mg_boost_emulator.git
cd mg_boost_emulator
```
Install dependencies:
```python
pip install -r requirements.txt
```
### Running the Demo

From the repo root:
```python
python -m examples.demo
```
This will:

Load the NN + GP emulators
Compute the MG boost
Plot the result
