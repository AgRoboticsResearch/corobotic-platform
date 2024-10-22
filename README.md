# Dependency
```bash
pip install -r requirements.txt
```
# Paper 
[A robotic orchard platform increases harvest throughput by controlling worker vertical positioning and platform speed](https://www.sciencedirect.com/science/article/pii/S0168169924001261)  
*Fei, Z., & Vougioukas, S. G. (2024). A robotic orchard platform increases harvest throughput by controlling worker vertical positioning and platform speed. Computers and Electronics in Agriculture, 218, 108735.*

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

# Four operational modes running at the same fruit distribution
Mode 1: Height and speed fixed mode 

![alt text](./img/mode1.gif)

Mode 2: Speed-optimized mode 

![alt text](./img/mode2.gif)

Mode 3: Height-optimized mode 

![alt text](./img/mode3.gif)

Mode 4: Full co-robotic mode 

![alt text](./img/mode4.gif)

# Convert image to gif to show the whole picking process
```bash
cd /results/modex/
convert *.png animation.gif
```
