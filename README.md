# Shape-Aware Oriented Bounding Box (OBB) to Horizontal Bounding Box (HBB) Conversion

## Abstract
---
Accurate object detection in aerial and satellite imagery is dependent upon the bounding box representation. This is especially true for spatially oriented objects such as ships or aircrafts. Oriented Bounding Boxes (OBB) have a tighter fit and more robust non-max suppression compared to Horizontal Bounding Boxes (HBB), any current post-processing conversion from OBB to HBB either introduces excess empty and background space or removes data from the detection. This paper introduces a novel approach for a shape-aware OBB-to-HBB conversion for ship detection in remote sensing imagery. It leverages hull shape, hull fullness, and the bounding box orientation to produce a tighter axis-aligned HBB representation. The proposed method is benchmarked against three baselines methods for OBB-to-HBB conversion, Outer HBB which uses minimum and maximum, Area Equivalent HBB and GBB Marginalized HBB.

## File Structure
---
There are 3 files present in this repository and are explained below : 
1. `obb2hbb.py` : This is the main file containing the code to convert provided OBB polygons to HBB polygons using the algorithm defined in the paper. There are three parameters which can be configured by the user, 
    1. fullness $\longrightarrow$ Scale factor for the width of the OBB. 
    2. q $\longrightarrow$ Shape exponent, defining the average shape of the hull, 
        1. q = 1 : Rhombus (diamond hull)
        2. q = 2 : Ellipse
        3. q $\rightarrow$ $\infty$ : Rectangle (Full OBB) 
    3. shrink $\longrightarrow$ Parameter to scale the HBB half-extents calculated by the algorithm.
2. `obbhbbstats.py` : This file is used for obtaining the mean Intersection Over Union (IoU), IoU between 0&#176;-15&#176; 15&#176;-30&#176; and 30&#176;-45&#176;, and undershoot and overshoot values. The values are obtained for the NovelShapeAware, OuterHBB, GBBMarginalised and AreaEquivalentHBB.
3. `visualise_methods.py` : This is only to see how the bounding boxes look after converting from OBB to HBB. It can also show the other three methods defined in the point above.

### Data Access
---
There are two datasets used with this paper,
1. [ShipRSImageNet](https://ui.adsabs.harvard.edu/abs/2021IJSTA..14.8458Z) : This is the primary dataset used for comparing the paper's proposed method with current industry standards, OuterHBB, AreaEquivalentHBB and GBBMarginalized. The dataset contains 2,745 images with multiple ships present in each image. This allowed the authours to see how well the proposed method performs. 
    
    The dataset is labeled using OBB polygons. These OBB polygons were converted using the above three methods and the paper's proposed method (NovelShapeAware). NovelShapeAware outperfomed the other three in terms of IoU and average error.

    The dataset can be found [here](https://universe.roboflow.com/palijo/shiprsimagenet_v1)

2. [S2 Custom Dataset](https://doi.org/10.5281/zenodo.20775561) : This dataset was created by the authours using Sentinel 2 data collected. It contains both HBB and OBB labels for ships present in the images. This dataset was used as a holdout set to see how does NovelShapeAware correspond with real world data.

### Usage

For getting the stats with `obbhbbstats.py`, there are three methods,
1. Using OBB labels only (DOTA Format) : `python obbhbbstats.py --mode novel_shape_aware --dataset_path <Path to OBB labels Folder> --output <Path to output JSON>`
2. Using OBB and HBB labels (DOTA Format) : `python obbhbbstats.py --mode novel_shape_aware --dataset_path <Path to OBB labels Folder> --output <Path to output JSON> --hbb_gt_labels <Path to HBB labels Folder>`
3. Using shapefiles for OBB and HBB : `python obbhbbstats.py --mode novel_shape_aware_gt --hbb_gt_root <Path to folder containing HBB shapefiles> --obb_gt_root <Path to folder containing OBB shapefiles> --output <Path to output JSON>`

For visualising the different methods, use the following command,
```bash
  python visualise_methods.py \
      --tif_roots  <Path to folder containing TIF files>
      --obb_gt_root <Path to folder containing OBB GT shapefiles> \
      --hbb_gt_root <Path to folder containing HBB GT shapefiles> \
      --output_dir  <Path to output directory for storing image chips with bounding boxes drawn>
```

### Requirements

The authours implemented the above code in Python 3.13.4. The requirements file is provided and can be installed using `pip install -r requirements.txt`

## Authors
1. **Badha Rathna Sabhapathy** : 
    Email : badharathna@gmail.com
    ORCID : 0009-0008-4672-1993
2. **Gotam Dahiya** : 
    Email : gotam.dahiya@skyserve.ai
    ORCID : 0009-0009-7440-5813
3. **Vishesh Vatsal** : 
    Email : vishesh@dfy-graviti.com
    ORCID : 0009-0008-2701-0010