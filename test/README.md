# Test Module
* IOU
* Accuracy
* Missing Rate

## Setup

Please put the generated maps that you want to test in the `./global_maps` directory. Please put ground truth maps  (`bev-5cm-crosswalks.jpg`, `bev-5cm-road.jpg`) in current directory i.e `./`. 

The file tree should look like this:

```bash
.
├── README.md
├── ground_truth
│   └── bev-5cm-crosswalks.jpg
│   └── bev-5cm-road.jpg
│   └── truth.npy (if does not exist, this will be generated from the jpg files)
├── global_maps
│   └── global_map_0721_new_filter.png
├── test_semantic_mapping.py
```

## Usage

* Testing

  Please run the following command to output testing results.

  ```bash
  python3 test_semantic_mapping.py
  ```

* Visualization

  To verify if the maps align, users can visalize two maps. Please pass `visualize=True` to `test.full_test` method. It will plot the ground truth map (left) and generated maps (right) with the same kinds of labels.

  ```bash
  python3 test_semantic_mapping.py -v
  ```

  

