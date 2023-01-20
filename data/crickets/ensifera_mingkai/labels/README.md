# Label IDs
0 (implied) = ambient
1 = echeme
2 = call -- multiple echemnes in a sequence, but not quite as fast as a trill
3 = trill
4 = noise

# Label Format
TSV (Tab-separated) files with lines in the following format:

`START_TIME    STOP_TIME    LABEL_ID`


# Notes
## General
* Time ranges for labelled ECHEMES might overlap with CALLS, i.e. they get counted twice.
* I sometimes used CALL or TRILL as a shortcut when there were too many single ECHEMES to label manually. My apologies for conflating the terminology in my simplified labelling scheme.
* Some files contain NOISE labels, but I did not exhaustively label all noise that I heard.

## Audio Specifics
* ZOOM003{8,9} contain other animal noises (unlabeled)
* SMA05027\_20220320\_053430 has some periodic background noise (maybe a fan?)
* SMA05089\_20220927\_091042 misc. noise and human speech (maybe TV?) throughout (unlabelled)
* SMA05027\_20220418\_072452 has quite a bit of noise and other animal calls
