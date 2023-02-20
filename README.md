# MetaSyn
Meta learning on continuous synthetic space


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
