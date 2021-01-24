import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import re
import sys
import random
import argparse
import shutil
import massedit

# read arguments
parser = argparse.ArgumentParser(description='Prepare data for ML experiments.')
parser.add_argument('--dataset', type=str, default="./dataset.pkl")
parser.add_argument('--exppath', type=str, default="./test_experiment")
parser.add_argument('--subsampling', type=int, default=10)
parser.add_argument('--clipsize', type=int, default=30)
poseFormat = 'videopose3d'
overlapSize = 20 
args = parser.parse_args()

fullVideoIds = [7, 16, 93, 136, 163, 215]

def sublist(lst1, lst2):
   ls2 = [element for element in lst2 if element not in lst1]
   return len(ls2) == 0

def splitSequence(seq, group_size, overlap_size):
    n = group_size
    m = overlap_size
    splits = [seq[i:i+n] for i in range(0, len(seq), n-m)]
    # remove splits that are only subsplits at the end
    # eg. [5,6,7,8], [6,7,8], [7,8], [8] --> last three should not be there
    while sublist(splits[-2], splits[-1]):
        splits.pop()

    return splits

def convert17to25(p17, frameInstances):
    conversion = [  (0, 9), (1, 8), (2, 14), (3, 15), (4, 16), (5, 11), (6, 12), (7, 13),
                    (8, 0), (9, 1), (10, 2), (11, 3), (12, 4), (13, 5), (14, 6), 
                    (15, -1), (16, -1), (17, -1), (18, -1), (19, -1), (20, -1), (21, -1), (22, -1), (23, -1), (24, -1) ]

    aux = np.zeros((25,frameInstances), dtype=np.float32)
    for (a,b) in conversion:
        if b != -1:
            aux[a,0] = p17[b]
    return aux

def convert17to18(p17, frameInstances):
    aux = np.zeros((18,frameInstances), dtype=np.float32)
    for idx in range(17):
        aux[idx,0] = p17[idx]
    aux[17,0] = 0
    return aux

def splitVideos(sequences, subSampling, groupSize, overlapSize):
    splits = []
    labels = []

    for seq in sequences:
        print('Processing', seq['info']['speakerId'])

        frames = seq['frames']
        gt = seq['attributes']

        if subSampling > 1:
            poses = frames[::subSampling]
        else:
            poses = frames 

        seqsplits = splitSequence(poses, group_size=groupSize, overlap_size=overlapSize)
        splits.extend( seqsplits )
        
        npgt = np.asarray([gt[attr] for attr in list(gt.keys())], dtype=np.float32)
        gts = np.stack([npgt for _ in range(len(seqsplits))], axis = 0)
        labels.extend(gts)

    return splits, np.asarray(labels)


def convertSplits(poseFormat, splits, groupSize, frameInstances=1):
    if poseFormat == 'videopose3d':
        numJoints = 18

    out = np.zeros((len(splits),3,groupSize,numJoints,frameInstances),dtype=np.float32)
    splitIdx = 0
    for split in splits:
        poseIdx = 0
        for pose in split:
            pose = np.asarray(pose)
            for coord in range(3):
                if poseFormat == 'videopose3d':
                    out[splitIdx][coord][poseIdx] = convert17to18(pose[:,coord], frameInstances)
                elif poseFormat == 'ntu':
                    out[splitIdx][coord][poseIdx] = convert17to25(pose[:,coord], frameInstances)
                else:
                    print("Unknown pose format", poseFormat)
                    sys.exit(1)

            poseIdx = poseIdx + 1

        splitIdx = splitIdx + 1

    return out

def shuffleData(data, labels, seed=13):
    print("shufflig", data.shape, labels.shape)
    # zip poses and labels for shuffling
    aux = list(zip(data, labels))
    random.seed(seed)
    random.shuffle(aux)
    # undo zip
    data, labels = zip(*aux)
    return list(data), list(labels)


def partitionData(clips, labels):
    # split into train/val/test
    trainPercentage = .8
    valPercentage = .3
    numVectors = len(clips)

    lenTrain = int(numVectors * trainPercentage)
    print("Taking", lenTrain, "samples for training and", numVectors - lenTrain, "samples for testing")

    # TRAIN
    lenTrainTrain = int(lenTrain * (1. - valPercentage))
    train = clips[:lenTrainTrain]
    trainLabels = labels[:lenTrainTrain]

    # VALIDATION
    val = clips[lenTrainTrain:lenTrain]
    valLabels = labels[lenTrainTrain:lenTrain]

    # TEST
    test = clips[lenTrain:]
    testLabels = labels[lenTrain:]

    return train, val, test, trainLabels, valLabels, testLabels

def write(data, rootPath, dataType, partition, speaker, poseFormat):

    outputType = 'GNN'

    if dataType == "skeletons" or dataType == "vectors":
        base = outputType + "_" + dataType + "_" + partition + "_" + speaker + "_" + poseFormat + ".npy"
        filename = os.path.join( rootPath, base )
        np.save(filename, data)
    else:
        aux = ( [""] * len(data), data )
        base = outputType + "_" + dataType + "_" + partition + "_" + speaker + "_" + poseFormat + ".pkl"
        filename = os.path.join( rootPath, base )
        f = open(filename, 'wb')
        pickle.dump(aux, f)
        f.close()

    print("Written", filename)
    return base

def prepareData(experiment, poseFormat, subSampling, groupSize, overlapSize):
    experimentPath = args.exppath
    print("Using experiment path:", experimentPath)

    if not os.path.isdir(experimentPath):
        os.mkdir(experimentPath)

    removeFullVideoSpeakers = True
    writeFiles = True 
    plotPartitionDistribution = False 

    f = open(args.dataset, 'rb')
    data = pickle.load(f)
    f.close()

    normalSpeakers = data['sequences']
    if len(normalSpeakers) == 0:
        print("No sequences!!!")
        return
    

    fullVideoSpeakers = [s for s in normalSpeakers if s['info']['speakerId'] in fullVideoIds]
    if removeFullVideoSpeakers:
        normalSpeakers = [s for s in normalSpeakers if s['info']['speakerId'] not in fullVideoIds]
        print("Removing full test video with ids", fullVideoIds, 'and', len(normalSpeakers), "videos after removal")

    # ALL SPEAKERS
    clips, labels = splitVideos(normalSpeakers, subSampling, groupSize, overlapSize)
    clips = convertSplits(poseFormat, clips, groupSize)
    shuffledClips, shuffledLabels = shuffleData(clips, labels)
    train, val, test, trainLabels, valLabels, testLabels = partitionData(shuffledClips, shuffledLabels)

    if plotPartitionDistribution:
        gtAttrNames = list(normalSpeakers[0]['attributes'].keys())
        _, axs = plt.subplots(3,6)
        xidx = 0
        for title, data in [('train', trainLabels), ('val',valLabels), ('test',testLabels)]:
            d = np.asarray(data)
            for attr, yidx in zip(gtAttrNames, range(6)):
                axs[xidx,yidx].hist(d[:, yidx], bins=60)
                axs[xidx,yidx].set(ylabel=title, xlabel=attr)
            xidx += 1

        plt.show()

    if writeFiles:
        trainskelsname = write(train, experimentPath, 'skeletons', 'train', 'all', 'videopose3d')
        trainlabelsname = write(trainLabels, experimentPath, 'labels', 'train', 'all', 'videopose3d')

        valskelsname = write(val, experimentPath, 'skeletons', 'val', 'all', 'videopose3d')
        vallabelsname = write(valLabels, experimentPath, 'labels', 'val' ,'all', 'videopose3d')

        testskelsname = write(test, experimentPath, 'skeletons', 'test', 'all', 'videopose3d')
        testlabelsname = write(testLabels, experimentPath, 'labels', 'test', 'all', 'videopose3d')

        destfilename = os.path.join(experimentPath, 'test.yaml')
        shutil.copy('./templates/test.yaml', destfilename )
        massedit.edit_files([destfilename], ["re.sub('--EXPPATH--', '" + re.escape(experimentPath) + "' , line)"], dry_run=False)
        massedit.edit_files([destfilename], ["re.sub('--TESTSKELSFILE--', '" + testskelsname + "' , line)"], dry_run=False)
        massedit.edit_files([destfilename], ["re.sub('--TESTLABELSFILE--', '" + testlabelsname + "' , line)"], dry_run=False)

        destfilename = os.path.join(experimentPath, 'train.yaml')
        shutil.copy('./templates/train.yaml', destfilename )
        massedit.edit_files([destfilename], ["re.sub('--EXPPATH--', '" + re.escape(experimentPath) + "' , line)"], dry_run=False)
        massedit.edit_files([destfilename], ["re.sub('--TRAINSKELSFILE--', '" + trainskelsname + "' , line)"], dry_run=False)
        massedit.edit_files([destfilename], ["re.sub('--TRAINLABELSFILE--', '" + trainlabelsname + "' , line)"], dry_run=False)
        massedit.edit_files([destfilename], ["re.sub('--VALSKELSFILE--', '" + valskelsname + "' , line)"], dry_run=False)
        massedit.edit_files([destfilename], ["re.sub('--VALLABELSFILE--', '" + vallabelsname + "' , line)"], dry_run=False)

    #FULL VIDEO TEST SPEAKERS
    for fvt in fullVideoSpeakers:
        speakerId = str(fvt['info']['speakerId'])
        clips, labels = splitVideos([fvt], subSampling, groupSize, overlapSize)
        clips = convertSplits(poseFormat, clips, groupSize)

        if writeFiles:
            testskelsname = write(clips, experimentPath, 'skeletons', 'test', 'id'+speakerId, 'videopose3d')
            testlabelname = write(labels, experimentPath, 'labels', 'test', 'id'+speakerId, 'videopose3d')

            destfilename = os.path.join(experimentPath, 'test_id' + speakerId + '.yaml')
            shutil.copy('./templates/test.yaml', destfilename )
            massedit.edit_files([destfilename], ["re.sub('--EXPPATH--', '" + re.escape(experimentPath) + "' , line)"], dry_run=False)
            massedit.edit_files([destfilename], ["re.sub('--TESTSKELSFILE--', '" + testskelsname + "' , line)"], dry_run=False)
            massedit.edit_files([destfilename], ["re.sub('--TESTLABELSFILE--', '" + testlabelsname + "' , line)"], dry_run=False)



prepareData(args.exppath, poseFormat, args.subsampling, args.clipsize, overlapSize)

