# MAC: Domain Generalization via Meta-learning on Anatomy Consistent Synthetic Representation 

<p align="center">
  <img src="/assets/overall.png" width="600" />
</p>

### Dirichlet Mixup
<p align="center">
  <img src="/assets/α = (5, 5, 5).png" width="150" />
  <img src="/assets/α = (1.5, 5, 5).png" width="150" />
  <img src="/assets/α = (1.5, 5, 1.5).png" width="150" /> 
  <img src="/assets/α = (4, 2, 2).png" width="150" />
</p>

### Data arrangement
Each dataset folder is arranged in the following format. And then paired and normalized by ```data_read.py```. Note that all the images are normalized into range [0,1] and saved in float32. 
```
|--Dataset_01
   |--img
      |--img01.tif
      |--img02.tif
   |--gt
      |--label01.png
      |--label02.png
   |--mask
      |--mask01.png
      |--mask02.png
|--Dataset_02      
```
