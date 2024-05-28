Welcome to the repository of my master thesis where all the files concerning my master thesis can be found. This project aims to leverage the power of machine learning algorithms to accurately forecast properties of polymers, such as atomization energies, HOMO energies, densities, boiling points etc. By integrating data-driven models, we strive to enhance the efficiency of polymer recycling, providing valuable insights and information that can accelerate research and development in the polymer recycling industry science.
For every diffrent dataset that was handled durign my thesis, another repository can be found:

<details>
<summary>Dataset overview and description:</summary>

|Dataset|Description|
|---|---|
|[Khazana Polymers](https://github.com/WannesVerh/Master_Thesis/tree/main/Khazana_dataset)|The Khazana dataset was mainly used to explore the possibilities within the realm of featurizers and algorithms. Since there is a constantly growing number of algorithms and features available, the focus will be laid upon the ones that can be easily integrated with the DeepChem framework|
|[HOPV/QM9](https://github.com/WannesVerh/Master_Thesis/tree/main/HOPV_QM9_dataset)|Both the HOPV and QM9 datsets holds values on the HOMO energies of various compounds. Because the QM9 dataset holds 134 thousand molecules the initial steps of building the model where done on the HOPV dataset, which only holds 350 molecules.|
|[Polyol Mixtures](https://github.com/WannesVerh/Master_Thesis/tree/main/Polyol_mixtures)|An attempt was made to create a dataset ourselfs which contained the densities of various mixtures of polyethylene-glycol and polypropylene-glycol polymers.|
|[Alkane Dataset](https://github.com/WannesVerh/Master_Thesis/tree/main/Alkane_dataset)|The dataset referred to as the “alkane dataset”, consists of various hydrocarbons, mostly alkanes but it also contains alkenes and alkynes. This dataset was build using the engineering software Aspen, which holds extensive databases comprising diverse molecules together with their associated properties. From the 32 properties available, seven were chosen based on their usefulness in the simulation of separation processes, namely: Critical pressure (Bar), Liquid molar volume at 25°C (m3/Kmol), Specific gravitation (au), Boiling temperature (°C), Critical temperature (°C), Critical compressibility factor (a.u.)|
|[Temperature dependent density](https://github.com/WannesVerh/Master_Thesis/tree/main/T_dependent_density_database)|This database contains the densities of the same alkanes as in the alkane datset. The desnities of the alkanes are simulated by the engineering software Aspen at various temperatures, ranging from 25°C to 500°C in 25°C intervals. This dataset was used to check the ability of machinle learning models to predict temperature dependent densities of various hydrocarbons.|




</details>


 These dataset repositories can conatin things like:
- **CSV-Files:** These repositories often contain the csv-files used to train the models, but also the raw data files collected from the online databases, etc...
- **Data handling files:** Files to handle the data and to get raw datafiles into usable formats.
- **Predicting scripts:** The scripts used to create/optimize the models.

![STRUCCHEM Logo](https://github.com/WannesVerh/Master_Thesis/blob/main/STRUCCHEM_logo.jpg)
