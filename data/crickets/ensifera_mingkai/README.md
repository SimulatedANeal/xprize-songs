# Label IDs
0 (implied) = ambient

1 = syllable

2 = echeme

3 = trill

4 = echeme sequence, call

5 = noise

6 = unknown other species

In some cases where I could more specifically identify the noise, I use a text label, e.g. "bird" or "phone"

# Label Format
TSV (Tab-separated) files with lines in the following format:

`START_TIME    STOP_TIME    LABEL_ID`

Times are in seconds. 

Filenames (minus extension) match the filename of its corresponding audio file
in the corresponding species directory of the same name.

# Test files
Test files for each species are specified in *split.py*

Test files were chosen based on a few criteria (in order of importance):
* Total example time should be ~10% of total data
* Some amount of noise that may better simulate data from contest environment
* A mixture of calls and ambient noise (silence) for better precision / recall representation.

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
