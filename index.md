

[YouTube Demo](https://www.youtube.com/watch?v=mS66yr1WmI4) | [Homepage](https://weijiawu.github.io/BOVText-Benchmark/)  |  [Github](https://github.com/weijiawu/BOVText-Benchmark) | Downloads(https://github.com/weijiawu/BOVText-Benchmark) | Paper(https://arxiv.org/abs/2112.04888) 

BOVText: A Large-Scale, Bilingual Open World Dataset for Video Text Spotting


Updated on June 06, 2021 (Added evaluation metric)

Released on May 26, 2021

## Description

We create a new large-scale benchmark dataset named **B**ilingual **O**pen **W**orld Video Text(BOVText), the first large-scale and Bilingual benchmark for video text spotting in a variety of scenarios.

The demo video can be found on [YouTube](https://www.youtube.com/watch?v=mS66yr1WmI4)

There are mainly three features for MMVText:
-  **Large-Scale**: we provide 510 videos with more than 1,000,000 frame images, four times larger than the existing largest dataset for text in videos. 
-  **Open Scenario**:BOVText covers 30+ open categories with a wide selection of various scenarios, e.g., life vlog, sports news, automatic drive, cartoon, etc. Besides, caption text and scene text are separately tagged for the two different representational meanings in the video. The former represents more theme information, and the latter is the scene information.
-  **Bilingual**:BOVText provides Bilingual text annotation to promote multiple cultures live and communication.
<img src="fig1_min.jpg" width="100%" class="aligncenter">

## News

## Tasks and Metrics
The proposed BOVText support four task(text detection, recognition, tracking, spotting), but mainly includes two tasks: 
-  Video Frames Detection. 
-  Video Frames Recognition. 
-  Video Text Tracking. 
-  End to End Text Spotting in Videos. 

MOTP (Multiple Object Tracking Precision)[1], MOTA (Multiple Object Tracking Accuracy) and IDF1[3,4] as the three important metrics are used to evaluate task1 (text tracking) for MMVText.
In particular, we make use of the publicly available py-motmetrics library (https://github.com/cheind/py-motmetrics) for the establishment of the evaluation metric. 

Word recognition evaluation is case-insensitive, and accent-insensitive. 
The transcription '###' or "#1" is special, as it is used to define text areas that are unreadable. During the evaluation, such areas will not be taken into account: a method will not be penalised if it does not detect these words, while a method that detects them will not get any better score.

### Task 3 for Text Tracking Evaluation
The objective of this task is to obtain the location of words in the video in terms of their affine bounding boxes. The task requires that words are both localised correctly in every frame and tracked correctly over the video sequence.
Please output the json file as following:
```
Output
.
├-Cls10_Program_Cls10_Program_video11.json
│-Cls10_Program_Cls10_Program_video12.json
│-Cls10_Program_Cls10_Program_video13.json
├-Cls10_Program_Cls10_Program_video14.json
│-Cls10_Program_Cls10_Program_video15.json
│-Cls10_Program_Cls10_Program_video16.json
│-Cls11_Movie_Cls11_Movie_video17.json
│-Cls11_Movie_Cls11_Movie_video18.json
│-Cls11_Movie_Cls11_Movie_video19.json
│-Cls11_Movie_Cls11_Movie_video20.json
│-Cls11_Movie_Cls11_Movie_video21.json
│-...


```
And then ```cd Evaluation_Protocol/Task1_VideoTextTracking```,
run following script:
```
python evaluation.py --groundtruths ./Test/Annotation --tests ./output

```

### Task 4 for Text Spotting Evaluation
Please output the json file like task 3.

```cd Evaluation_Protocol/Task2_VideoTextSpotting```,
run following script:
```
python evaluation.py --groundtruths ./Test/Annotation --tests ./output

```

## Ground Truth (GT) Format and Downloads

We create a single JSON file for each video in the dataset to store the ground truth in a structured format, following the naming convention:
gt_[frame_id], where frame_id refers to the index of the video frame in the video

In a JSON file, each gt_[frame_id] corresponds to a list, where each line in the list correspond to one word in the image and gives its bounding box coordinates, transcription, text type(caption or scene text) and tracking ID, in the following format:

```
{

“frame_1”:  
            [
			{
				"points": [x1, y1, x2, y2, x3, y3, x4, y4],
				“tracking ID”: "1" ,
				“transcription”: "###",
				“category”: title/caption/scene text,
				“language”: Chinese/English,
				“ID_transcription“:  complete words for the whole trajectory
			},

               …

            {
				"points": [x1, y1, x2, y2, x3, y3, x4, y4],
				“tracking ID”: "#" ,
				“transcription”: "###",
				“category”: title/caption/scene text,
				“language”: Chinese/English,
				“ID_transcription“:  complete words for the whole trajectory
			}
			],

“frame_2”:  
            [
			{
				"points": [x1, y1, x2, y2, x3, y3, x4, y4],
				“tracking ID”: "1" ,
				“transcription”: "###",
				“category”: title/caption/scene text,
				“language”: Chinese/English,
				“ID_transcription“:  complete words for the whole trajectory
			},

               …

            {
				"points": [x1, y1, x2, y2, x3, y3, x4, y4],
				“tracking ID”: "#" ,
				“transcription”: "###",
				“category”: title/caption/scene text,
				“language”: Chinese/English,
				“ID_transcription“:  complete words for the whole trajectory
			}
			],

……

}
```


### Downloads
 
- The BOVText dataset is available for **non-commercial research purposes only**.
- Please download the [agreement](https://github.com/weijiawu/BOVText-Benchmark/tree/main/Dataset/bovtext_agreement.pdf) and read it carefully.
- Please ask your supervisor/advisor to sign the agreement appropriately and then send the scanned version (example) to Weijia Wu (weijiawu@zju.edu.cn).
- After verifying your request, we will contact you with the dataset download link.

## Maintenance Plan and Goal
The author will plays an active participant in the video text field and maintaining the dataset at least before 2023 years.
And the maintenance plan as the following: 
- [x] Merging and releasing the whole dataset after further review. (Around before November, 2021)
- [x] Updating evaluation guidance and script code for four tasks(detection, tracking, recognition, and spotting). (Around before November, 2021)
- [ ] Hosting a competition concerning our work for promotional and publicity. (Around before March,2022)

More video-and-language tasks will be supported in our dataset:
- [ ] Text-based Video Retrieval[5] (Around before March,2022)
- [ ] Text-based Video Caption[6] (Around before September,2023)
- [ ] Text-based VQA[7][8] (TED)


## Citation



```
@article{wu2021opentext,
  title={A Bilingual, OpenWorld Video Text Dataset and End-to-end Video Text Spotter with Transformer},
  author={Weijia Wu, Debing Zhang, Yuanqiang Cai, Sibo Wang, Jiahong Li, Zhuang Li, Yejun Tang, Hong Zhou},
  journal={35th Conference on Neural Information Processing Systems (NeurIPS 2021) Track on Datasets and Benchmarks},
  year={2021}
}
```

## Organization

Affiliations: [Zhejiang University](https://www.zju.edu.cn/english/), [MMU of Kuaishou Technology](https://www.kuaishou.com/en)

Authors: Weijia Wu(Zhejiang University), [Debing Zhang](https://scholar.google.com/citations?user=4nL1cDEAAAAJ&hl=en)(Kuaishou Technology)



## Feedback

Suggestions and opinions of this dataset (both positive and negative) are greatly welcome. Please contact the authors by sending email to
`weijiawu@zju.edu.cn`.

## License and Copyright
The project is open source under CC-by 4.0 license (see the ``` LICENSE ``` file).

Only for research purpose usage, it is not allowed for commercial purpose usage.

The videos were partially downloaded from YouTube and some may be subject to copyright. We don't own the copyright of those videos and only provide them for non-commercial research purposes only.
For each video from YouTube, while we tried to identify video that are licensed under a Creative Commons Attribution license, we make no representations or warranties regarding the license status of each video and you should verify the license for each image yourself.

Except where otherwise noted, content on this site is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/).



## References

[1] Dendorfer, P., Rezatofighi, H., Milan, A., Shi, J., Cremers, D., Reid, I., Roth, S., Schindler, K., & Leal-Taixe, L. (2019). CVPR19 Tracking and Detection Challenge: How crowded can it get?. arXiv preprint arXiv:1906.04567.

[2] Bernardin, K. &amp; Stiefelhagen, R. Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. Image and Video Processing, 2008(1):1-10, 2008.

[3] Ristani, E., Solera, F., Zou, R., Cucchiara, R. & Tomasi, C. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. In ECCV workshop on Benchmarking Multi-Target Tracking, 2016.

[4] Li, Y., Huang, C. &amp; Nevatia, R. Learning to associate: HybridBoosted multi-target tracker for crowded scene. In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2009.



