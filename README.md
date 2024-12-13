<div align="center">
<h1>TODO:</h1>

[Vladislav Meshkov]()<sup>1, 2 :email:</sup>, [Andrey Grabovoy](https://github.com/andriygav)<sup>1</sup>

<sup>1</sup> Moscow Institute of Physics and Technology

<sup>:email:</sup> Corresponding author

[📝 Paper]()
</div>

**Installation:**  

- git clone git@github.com:Drago160/Meshkov-paper.git
- pip install requirements.txt

**Experiments:**

Choose any config you need and train appropriate group of models, for exmple:
- python3 train --config_path=configs/layers_num/fashion_mnist.yml

Visualize it, using
- python3 visualize_results.py --config_path=config/layers_num/fashion_mnist.yml


Results will be in figures/fashion_mnist_change_layers.pdf
