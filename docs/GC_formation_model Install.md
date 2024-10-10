Package install instruction for GC_formation_model by Yintiang (Bill) Chen & Oleg Gnedin 
https://github.com/ybillchen/GC_formation_model

1. Clone package into convenient location (not within working directory)
2. In working directory activate virtual environment
3. Make sure `setuptools` is up to date (https://stackoverflow.com/questions/50585246/pip-install-creates-only-the-dist-info-not-the-package). This can be done by: 
```
python3 -m pip install --upgrade pip setuptools wheel
```
4. cd into GC_formation_model 
```
pip3 install -e .
```
5. cd back into working directory 
6. Import model (note it may try and get you to install extra packages, do so)
7. Should now be able to import properly