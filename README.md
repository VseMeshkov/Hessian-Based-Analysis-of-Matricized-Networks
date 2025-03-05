<div align="center">
<h1>ConvNets Landscape Convergence: Hessian-Based
Analysis of Matricized Networks</h1>

[Vladislav Meshkov](), [Nikita Kiselev](), [Andrey Grabovoy](https://github.com/andriygav)

<sup>1</sup> Moscow Institute of Physics and Technology

[📝 Paper]()
</div>https://ieeexplore.ieee.org/document/10899113?denied=</div>

**Installation:**  

- git clone git@github.com:Drago160/Meshkov-paper.git
- pip install requirements.txt

**Experiments:**

Choose any config you need and train appropriate group of models, for exmple:
- python3 train --config_path=configs/layers_num/fashion_mnist.yml

Visualize it, using
- python3 visualize_results.py --config_path=config/layers_num/fashion_mnist.yml


Results will be in figures/fashion_mnist_change_layers.pdf

## Citation
```BibTeX
@INPROCEEDINGS{10899113,  
    author={Meshkov, Vladislav and Kiselev, Nikita and Grabovoy, Andrey},  
    booktitle={2024 Ivannikov Ispras Open Conference (ISPRAS)},  
    title={ConvNets Landscape Convergence: Hessian-Based Analysis of Matricized Networks},  
    year={2024},  
    volume={},  
    number={},  
    pages={1-10},  
    keywords={Geometry;Sensitivity;Artificial neural networks;Network architecture;Convergence},  
    doi={10.1109/ISPRAS64596.2024.10899113}  
} 
