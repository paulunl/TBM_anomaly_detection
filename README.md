#TBM_anomaly_detection

Code and data repository for the paper TBM Operational Data-Driven Anomaly Detection in Hard Rock Excavations by Paul Unterlaß1, Mario Wölflingseder1, Thomas Marcher1

    Institute of Rock Mechanics and Tunnelling, Graz University of Technology, Rechbauerstraße 12, Graz, Austria

    correspondence: unterlass@tugraz.at

Code authors: Paul Unterlass & Mario Wölflingseder

Synthetic TBM operational data

The synthetic Tunnel Boring Machine (TBM) operational data can be found in the folder "data". Datasets for 2 different TBMs are available, denoted as TBM A, -B. The data was synthezised using generative adverserial networks (GANs) based on real TBM operational data.
Further synthetic data and the code of GANs can be found in the following Github repository: https://github.com/geograz/TBM_advance_classification
Further information on the synthetic data in can be found in the following publications: https://doi.org/10.1007/s00603-025-04542-4 (open access) and https://doi.org/10.1007/978-3-031-20241-4_1

Requirements

The environment is set up using conda.

To do this create an environment called TBM_anomaly_detection using environment.yaml with the help of conda. If you get pip errors, install pip libraries manually, e.g. pip install pandas

conda env create --file environment.yaml

Activate the new environment with:

conda activate Jv

contact

unterlass@tugraz.at
