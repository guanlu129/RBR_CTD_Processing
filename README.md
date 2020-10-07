RBR_CTD_Processing

For processing raw RBR CTD data in Python.
RBR_Processing.py contains processing steps:
- Export raw data from .rsk to .csv files
- Create Metadata dictionay
- Prepare CTD_DATA.csv with 6 line headers for IOSShell
- Plot and check profile locations
- Plot and Check for zero-order hold
- CALIB: Pressure/Depth correction
- CLIP: remove measurements near sea surface and bottom
- FILTER: apply a low pass filter
- SHIFT: shift conductivity and recalculate salinity
- SHIFT: shift oxygen
- DELETE: remove the pressure reversal
- BINAVE: calculate bin averages
- EDIT: apply final editing
- Prepare .ctd files with IOS Header File

Reference:
Halverson, M., Jackson, J., Richards, C., Melling, H., Brunsting, R., Dempsey, M., Ga-
tien, G., Hamilton, A., Hunt, B., Jacob, W., and Zimmerman, S. 2017. Guidelines
for proces
