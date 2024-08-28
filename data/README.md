# Instances

All the instances can be grouped in four families:
- **BENG**: instances for 2D-BPP proposed in [*Packing rectangular pieces–A heuristic approach*](https://academic.oup.com/comjnl/article/25/3/353/369826).
- **thpack9**: instances for 3D-BPP proposed in [*An integer programming based heuristic approach to the three dimensional packing problem*](https://iaorifors.com/paper/4516).
- **dataset[X]**: our proposed realistic instances (note that the *dataset_small* is just a dummy instance for testing). 
- **MODdataset[X]**: modified version of our instances with only 50 items and reduced area of the vehicles (created with the `mod_dataset.py` in the *scripts* folder).
- **test_exact_[X]**: instances to be run with the exact solver, containing one truck and a small set of items.