=================================================
Introduction
=================================================

This is the documentation for the linux MegaFace Development Kit. The purpose
of the development kit is to provide you with scripts to evaluate performance of
trained models on our challenges. The evaluation is how well your algorithm
can identify matching probe images with multiple distractor images.

Table of Contents:
  1. Overview of included files
    1.1 Code
    1.2 Baseline models and template lists
    1.3 Usage
    1.4 Required Installation
  2. Datasets
    2.1 MegaFace
    2.2 MegaFace 2 Training Set
    2.3 MegaFace Disjoint Distractors
    2.4 FaceScrub
    2.5 FGNet
  3. Running Experiments
    3.1 Getting Started
    3.2 Run baseline
    3.3 Run with your features
    3.4 Process Overview
  4. Reporting results

Please contact us at contact.megaface@gmail.com for questions, comments,
or bug reports.

=================================================
1. Overview of included files
=================================================

1.1 Code:
    /experiments/run_experiment.py - Runs Identification experiment with different size distractor sets
    /scripts/matio.py - Utility functions for loading and saving features in the format we use

1.2 Baseline models and template lists (only needed for challenge 1 baseline code test):
    /models/jb_identity.bin - Scoring model used by default.
    /models/jb_LBP.bin - Scoring model for running Joint Bayes baseline test.

1.3 Challenge 2 distractor template lists
    /templatelists/challenge2

1.3 Usage
    run_experiment.py - <distractors_features_path> <probe_features_path> <out_root> optional: -d -s <sizes> -m <model> -ns <number of sets to run> -si <indices of sets to run>
        megaface_features_path - Path to root directory of megaface features (disjoint features if running challenge 2)
        probe_feature_list - Path to root directory of facescrub, or FGNet (probe) features
        out_root - File output directory, outputs score matrix files, feature lists, and results files
        -d Flag to delete intermediate probe-distractor score matrix files. Can be used in order to save space needed to run
        -s {size,...}, --sizes {size,...} - (optional) Size(s) of feature list(s) to create. Default: 10 100 1000 10000 100000 1000000
        -m {model filename}, --model {model filename} - (optional) Scoring model to use. Default: ../models/jb_identity.bin

        Files Output:
            Feature List Files - megaface_features_{group file ending}_{size}_{set}
            Probe-Distractor Score Matrix Files - facescrub_megaface_{group file ending}_{size}_{set}.bin
            Probe-Probe Score Matrix Files - facescrub_facescrub_{group file ending}.bin
            Results files - results/cmc_facescrub_megaface_{group file ending}_{size}_{set}.json
                      - results/matches_facescrub_megaface_{group file ending}_{size}_{set}.json

1.4 Required Installation:
    * OpenCV - Open source computer vision and machine learning software library
        (http://opencv.org/)

=================================================
2. Datasets
=================================================

2.1 MegaFace (Challenge 1)
    Dataset - Gallery dataset comprised of photos from Flickr users
        Download link is available on our website (http://megaface.cs.washington.edu/dataset/download.html)

2.2 Megaface 2 Training Set (Challenge 2)
    Dataset - Training faces organized by identity. Challenge 2 only allows training your model
    on this set.
    Download link is available on our website (http://megaface.cs.washington.edu/dataset/download_training.html)

2.3 MegaFace Disjoint (Challenge 2)
    Dataset - Gallery dataset which is entirely disjoint from our training set
        Download link is available on our website (http://megaface.cs.washington.edu/dataset/download_training.html)
        Template lists for this set are in /templatelists/challenge2

2.4 FaceScrub (Challenge 1 & 2)
    Dataset - Probe dataset comprised of celebrities
        Download links:
        - FaceScrub uncropped set (http://megaface.cs.washington.edu/dataset/download/content/downloaded.tgz)
        - FaceScrub cropped and json files (http://megaface.cs.washington.edu/dataset/download/content/test_cropped.zip)
        * You may use either the full set or the cropped set, but additional parameters will be needed if the uncropped set is used due to naming differences
        - FaceScrub bounding boxes for actors (http://megaface.cs.washington.edu/dataset/download/content/facescrub_actors.txt)
        - FaceScrub bounding boxes for actresses (http://megaface.cs.washington.edu/dataset/download/content/facescrub_actresses.txt)
        Template list for this data set located in /templatelists

2.5 FGNet (Challenge 1 & 2)
    Dataset - Probe dataset comprised of people at various ages
        Download links:
        - FGNet and json meta files (http://megaface.cs.washington.edu/dataset/download/content/FGNET2.tar.gz)
        Template list for this data set located in /templatelists



=================================================
3. Running Experiments
=================================================

3.1 Getting Started
    prereq: python 2.7, opencv, anaconda

    Feature Files Format:
    Feature files are expected to be formatted as OpenCV mat files with N rows x 1 column

3.2 Run Challenge 1 baseline:
    Download MegaFace feature files (as .zip or .tar.gz), e.g.,
        http://megaface.cs.washington.edu/dataset/download/content/MegaFace_Features.zip
    Download FaceScrub feature files:
        http://megaface.cs.washington.edu/dataset/download/content/FaceScrub_LBPFeatures.zip
    LBP:
    > python run_experiment.py ../../MegaFace/MegaFace_Features/ ../../MegaFace/FaceScrubSubset_Features/ _LBP_100x100.bin ../../MegaFace/LBP_results/
    Joint Bayes:
    > python run_experiment.py -m ../models/jb_LBP.bin ../../MegaFace/MegaFace_Features/ ../../MegaFace/FaceScrubSubset_Features/ _LBP_100x100_proj.bin ../../MegaFace/JB_results/

3.3 Run with your features (Challenge 1 & 2):
    For challenge 2 add -dlp templatelists/challenge2 to set distractor paths,
    as by default it will look for challenge1 distractors.

    Using cropped facescrub set:
    > python run_experiment.py ../../MegaFace/MegaFace_{algorithm name}Features ../../MegaFace/FaceScrub_{algorithm name}Features {features file ending} ../../MegaFace/results/

    Using FGNet set:
    > python run_experiment.py -p templatelists/fgnet_features_list.json ../../MegaFace/MegaFace_{algorithm name}Features ../../MegaFace/FaceScrub_{algorithm name}Features {features file ending} ../../MegaFace/results/

    Using uncropped facescrub set:
    > python run_experiment.py -p ../templatelists/facescrub_uncropped_features_list.json ../../MegaFace/MegaFace_{algorithm name}Features ../../MegaFace/FaceScrub_{algorithm name}Features {features file ending} ../../MegaFace/results/

    For fixed number of distractors (for debugging) use -s option:
    > python run_experiment.py -s 100 ../../MegaFace/MegaFace_{algorithm name}Features ../../MegaFace/FaceScrub_ {features file ending} ../../MegaFace/results/

3.4 Process Overview
    1. Produce features for megaface (disjoint if doing challenge 2),
       FGNet, and facescrub datasets
        - matio.py may be used for saving feature files
    2. Run run_experiment.py as specified above
           Be sure to set the -dlp distractor list option to the disjoint list if doing
           Challenge 2!
           Outputs used feature list files: megaface_features_{algorithm name}_{size}_{set #}
                                            facescrub_features_{algorithm name}
                                            fgnet_features_{algorithm name}
           Outputs score matrix files: facescrub_megaface_{algorithm name}_{distractor set size}_{set #}.bin
                                        facescrub_facescrub_{algorithm name}.bin
           Outputs results files: cmc_facescrub_megaface_{algorithm name}_{distractor set size}_{set #}.json
                                  matches_facescrub_megaface_{algorithm name}_{distractor set size}_{set #}.json
    3. Upload outputted results files for each distractor size as specified below
           * Please also input links to FaceScrub and MegaFace features

=================================================
4. Reporting results
=================================================

Please use the google drive folder you received with access information. This was created for you to upload your results into.
  Filename Formatting:
       CMC/ROC - cmc_facescrub_megaface_{algorithm name}_{distractor set size}_{set #}.json
       Matches - matches_facescrub_megaface_{algorithm name}_{distractor set size}_{set #}.json

Input contact information and other information about your group into the ContactInfo spreadsheet located in
the google drive folder created for you.
