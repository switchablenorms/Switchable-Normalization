# Created by MegaFace Team
# Please cite the our paper if you use our code, results, or dataset in a publication
# http://megaface.cs.washington.edu/ 

import argparse
import os
import sys
import json
import random
import subprocess

MODEL = os.path.join('..', 'models', 'jb_identity.bin')
IDENTIFICATION_EXE = os.path.join('..', 'bin', 'Identification')
FUSE_RESULTS_EXE = os.path.join('..', 'bin', 'FuseResults')
MEGAFACE_LIST_BASENAME = os.path.join('..','templatelists','megaface_features_list.json')
PROBE_LIST_BASENAME = os.path.join('..','templatelists','facescrub_features_list.json')

def main():
    parser = argparse.ArgumentParser(description=
        'Runs the MegaFace challenge experiment with the provided feature files')
    parser.add_argument('distractor_feature_path', help='Path to MegaFace Features')
    parser.add_argument('probe_feature_path', help='Path to FaceScrub Features')
    parser.add_argument('file_ending',help='Ending appended to original photo files. i.e. 11084833664_0.jpg_LBP_100x100.bin => _LBP_100x100.bin')
    parser.add_argument(
        'out_root', help='File output directory, outputs results files, score matrix files, and feature lists used')
    parser.add_argument('-s', '--sizes', type=int, nargs='+',
                        help='(optional) Size(s) of feature list(s) to create. Default: 10 100 1000 10000 100000 1000000')
    parser.add_argument('-m', '--model', type=str,
                        help='(optional) Scoring model to use. Default: ../models/jb_identity.bin')
    parser.add_argument('-ns','--num_sets', help='Set to change number of sets to run on. Default: 1')
    parser.add_argument('-d','--delete_matrices', dest='delete_matrices', action='store_true', help='Deletes matrices used while computing results. Reduces space needed to run test.')
    parser.add_argument('-p','--probe_list', help='Set to use different probe list. Default: ../templatelists/facescrub_features_list.json')
    parser.add_argument('-dlp','--distractor_list_path', help='Set to change path used for distractor lists')
    parser.set_defaults(model=MODEL, num_sets=1, sizes=[10, 100, 1000, 10000, 100000, 1000000], probe_list=PROBE_LIST_BASENAME, distractor_list_path=os.path.dirname(MEGAFACE_LIST_BASENAME))
    args = parser.parse_args()

    distractor_feature_path = args.distractor_feature_path
    out_root = args.out_root
    probe_feature_path = args.probe_feature_path
    model = args.model
    num_sets = args.num_sets
    sizes = args.sizes
    file_ending = args.file_ending
    alg_name = file_ending.split('.')[0].strip('_')
    delete_matrices = args.delete_matrices
    probe_list_basename = args.probe_list
    megaface_list_basename = os.path.join(args.distractor_list_path,os.path.basename(MEGAFACE_LIST_BASENAME))
    set_indices = range(1,int(num_sets) + 1)

    assert os.path.exists(distractor_feature_path)
    assert os.path.exists(probe_feature_path)
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    if(not os.path.exists(os.path.join(out_root, "otherFiles"))):
        os.makedirs(os.path.join(out_root, "otherFiles"))
    other_out_root = os.path.join(out_root, "otherFiles")

    probe_name = os.path.basename(probe_list_basename).split('_')[0]
    distractor_name = os.path.basename(megaface_list_basename).split('_')[0]

    #Create feature lists for megaface for all sets and sizes and verifies all features exist
    missing = False
    for index in set_indices:
        for size in sizes:
            print('Creating feature list of {} photos for set {}'.format(size,str(index)))
            cur_list_name = megaface_list_basename + "_{}_{}".format(str(size), str(index))
            with open(cur_list_name) as fp:
                featureFile = json.load(fp)
                path_list = featureFile["path"]
                for i in range(len(path_list)):
                    path_list[i] = os.path.join(distractor_feature_path,path_list[i] + file_ending)
                    if(not os.path.isfile(path_list[i])):
                        print (path_list[i] + " is missing")
                        missing = True
                    if (i % 10000 == 0 and i > 0):
                        print( str(i) + " / " + str(len(path_list)))
                featureFile["path"] = path_list
                json.dump(featureFile, open(os.path.join(
                    other_out_root, '{}_features_{}_{}_{}'.format(distractor_name,alg_name,size,index)), 'w'), sort_keys=True, indent=4)
    if(missing):
        sys.exit("Features are missing...")
    
    #Create feature list for probe set
    with open(probe_list_basename) as fp:
        featureFile = json.load(fp)
        path_list = featureFile["path"]
        for i in range(len(path_list)):
            path_list[i] = os.path.join(probe_feature_path,path_list[i] + file_ending)
            if(not os.path.isfile(path_list[i])):
                print (path_list[i] + " is missing")
                missing = True
        featureFile["path"] = path_list
        json.dump(featureFile, open(os.path.join(
            other_out_root, '{}_features_{}'.format(probe_name,alg_name)), 'w'), sort_keys=True, indent=4)
        probe_feature_list = os.path.join(other_out_root, '{}_features_{}'.format(probe_name,alg_name))
    if(missing):
        sys.exit("Features are missing...")

    print('Running probe to probe comparison')
    probe_score_filename = os.path.join(
        other_out_root, '{}_{}_{}.bin'.format(probe_name, probe_name, alg_name))
    proc = subprocess.Popen(
        [IDENTIFICATION_EXE, model, "path", probe_feature_list, probe_feature_list, probe_score_filename])
    proc.communicate()

    for index in set_indices:
        for size in sizes:
            print('Running test with size {} images for set {}'.format(
                str(size), str(index)))
            args = [IDENTIFICATION_EXE, model, "path", os.path.join(other_out_root, '{}_features_{}_{}_{}'.format(distractor_name,alg_name,size,index)
                ), probe_feature_list, os.path.join(other_out_root, '{}_{}_{}_{}_{}.bin'.format(probe_name, distractor_name, alg_name, str(size),str(index)))]
            proc = subprocess.Popen(args)
            proc.communicate()

            print('Computing test results with {} images for set {}'.format(
                str(size), str(index)))
            args = [FUSE_RESULTS_EXE]
            args += [os.path.join(other_out_root, '{}_{}_{}_{}_{}.bin'.format(
                probe_name, distractor_name, alg_name, str(size), str(index)))]
            args += [os.path.join(other_out_root, '{}_{}_{}.bin'.format(
                probe_name, probe_name, alg_name)), probe_feature_list, str(size)]
            args += [os.path.join(out_root, "cmc_{}_{}_{}_{}_{}.json".format(
                probe_name, distractor_name, alg_name, str(size), str(index)))]
            args += [os.path.join(out_root, "matches_{}_{}_{}_{}_{}.json".format(
                probe_name, distractor_name, alg_name, str(size), str(index)))]
            proc = subprocess.Popen(args)
            proc.communicate()

            if(delete_matrices):
                os.remove(os.path.join(other_out_root, '{}_{}_{}_{}_{}.bin'.format(
                    probe_name, distractor_name, alg_name, str(size), str(index))))

if __name__ == '__main__':
    main()
