python3 GenData.py --dataset [dataset-name]
    - dataset-name:
        - syn1, syn2, ... , syn6
        - bitcoinalpha
        - bitcoinotc
    - Generate feature matrix X, adjacency matrix A and ground-truth label L into "XAL" folder
   
python3 GenGroundTruth.py --dataset [dataset-name]
    - dataset-name:
        - bitcoinalpha
        - bitcoinotc
    - Generate Explanations in "ground_truth_explanation" folder
   