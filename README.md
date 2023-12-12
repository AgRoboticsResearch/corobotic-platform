# Dependency
```bash
pip install -r requirements.txt
```

# Compile simulator
Some cython extensions are used by simulator so need to compile them
``` bash
cd simulator
python setup.py build_ext --inplace
``` 

# Demo 
Check demo.ipynb to play with 4 platform operation modes

# Import simulator
```python
from simulator.appleSimulator import appleSimulator
```

# Convert image to gif to show the whole picking process
```bash
cd /results/modex/
convert *.png animation.gif
```
