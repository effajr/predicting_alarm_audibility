# data
This folder contains the audio files corresponding to the alarms and backgrounds, along with the perceptual annotations.

The content of this folder can be downloaded *via* this [link](https://zenodo.org/records/11353196/files/data.zip?download=1) and should be unzipped here.

When the content is unzipped, features can be extracted by running [extract_features.py](../prepare_data/extract_features.py).

## Structure

```
data/
├─  annotations/
│  ├─ dev/
│  │  ├─ annotation_compilation_dev.csv : Compilation of all the listening conditions and 
│  │  │                                   individual annotator responses for the development data.
│  │  ├─ dev_conditions.csv : Unique listening conditions (extracted from annotation_compilation_dev.csv).
│  │  ├─ dev_labels.csv : All individual annotator responses for each 
│  │  │                   unique listening condition (extracted from annotation_compilation_dev.csv).
│  │  ├─ dev_train_valid_split.csv : Random 80%/20% training/validation split used for development 
│  │  │                              in the experiments reported in the paper. 
│  ├─ eval/
│  │  ├─ annotation_compilation_eval.csv :  Compilation of all the listening conditions and individual annotator 
│  │  │                                     responses for the evaluation data.
│  │  │                                     The column 'clearly_audible_mean' represents individual annotator 
│  │  │                                     binary responses evaluated for each listening condition.
│  │  │                                     The column 'clearly_audible_pf' represents individual annotator 
│  │  │                                     psychometric functions evaluated for each listening condition.
│  │  │
│  │  ├─ eval_conditions.csv : Unique listening conditions (extracted from annotation_compilation_eval.csv).
│  │  ├─ eval_labels_apf.csv : All individual annotator psychometric function values for each 
│  │  │                        unique listening condition (extracted from annotation_compilation_eval.csv).
│  │  ├─ eval_labels_mv.csv : All individual annotator binary responses for each 
│  │  │                       unique listening condition (extracted from annotation_compilation_eval.csv)
│
├─  audio/ : .wav files corresponding to the alarms and backgrounds for Development and 
│  │         Evaluation subsets of the dataset.
│  ├─ dev/
│  │  ├─ alarms/
│  │  ├─ backgrounds/
│  ├─ eval/
│  │  ├─ alarms/
│  │  ├─ backgrounds/
```
