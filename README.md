
# Heliophysics SPAcecraft Conjunction timEseries analYsis (HelioSPACEY)

This project aims to facilitate the analysis of timeseries data obtained by space-based instruments from heliophysics missions. 
It was developed for the purpose of [finding conjunctions between spacecraft from in-situ data](https://meetingorganizer.copernicus.org/EGU24/EGU24-12885.html). The missions currently supported by this project are Solar Orbiter, Parker Solar Probe, BepiColombo and STEREO-A. The project enables the use of magnetohydrodynamic (MHD) [ENLIL simulations](https://ccmc.gsfc.nasa.gov/models/CORHEL-MAS_WSA_ENLIL~5.0/) as input to generate synthetic data for the training of machine learning (ML) models.

HelioSPACEY enables:
- extraction of synthetic data from [simulations](https://ccmc.gsfc.nasa.gov/models/CORHEL-MAS_WSA_ENLIL~5.0/) by flying 'virtual spacecraft' through the simulation space,
- querying of timeseries data obtained from [curated data products](https://omniweb.gsfc.nasa.gov/coho/html/cw_data.html) from heliophysics missions,
- processing and visualisation of these timeseries,
- feature engineering from these timeseries for the purpose of training machine learning models,
- integration with popular Python machine learning libraries,
- identification of conjunctions between spacecraft resulting in linked in-situ observations relevant to the characterisation of solar wind expansion.

Main contributors to this project include: Zoe Faes, Laura Hayes, Andrew Walsh and Daniel Müller.

This project relies heavily on Sunpy, Astropy, astrospice, Numpy, Pandas, Matplotlib, scikit-learn and PyTorch.
