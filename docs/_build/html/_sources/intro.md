# COMP5900 Mouse IRL
Using Inverse Reinforcement Learning to infer the reward functions of mice in a popular behavioural paradigm (Open Field Test)
![](demo.gif)


```{tableofcontents}
```

## Setup
```
git clone git@github.com:A-Telfer/mouse-irl.git
cd mouse-irl
pip install .
python demo.py
```

## Building documentation
Install tpols with
```
pip install jupyter-book ghp-import
```

Then build changes
```
jb build docs
```

Finally upldate
```
ghp-import -n -p -f docs/_build/html
```
