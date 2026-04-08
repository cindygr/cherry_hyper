winget install astral-sh.uv -h --accept-source-agreements
uv python install 3.12.7
uv python pin 3.12.7
uv init --bare
uv add numpy scipy matplotlib ipykernel ipympl spectral rasterio xmltodict imageio pip torch torchvision
uv add skimage sklearn