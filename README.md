<div align="center">
  <img src="Dataset/image/logo.png" width="500"/>
</div>


MOVText: A Large-Scale, **M**ultilingual **O**pen World Dataset for **V**ideo Text Spotting


Updated on June 06, 2021 (Added evaluation metric)

Released on May 26, 2021

<img src="Dataset/image/demo.gif" width="400"/>  <img src="Dataset/image/demo1.gif" width="400"/>


## Description
[YouTube Demo](https://www.youtube.com/watch?v=mS66yr1WmI4) | [Homepage](https://weijiawu.github.io/MMVText-Benchmark/)  |  Downloads(TED) | Paper(TED) 

(Note: a small part(46 videos) of training set can be found in [Baidu Cloud](https://pan.baidu.com/s/1wJDVS_fSqP0jXnVFYP4TxQ)
password: woa8) for reference, the whole dataset is currently under merging and would be released before October 15.)

We create a new large-scale benchmark dataset named **M**ultilingual, **O**pen World **V**ideo Text(MOVText), the first large-scale and multilingual benchmark for video text spotting in a variety of scenarios.
All data are collected from [KuaiShou](https://www.kuaishou.com/en) and [YouTube](https://www.youtube.com/)

There are mainly three features for MOVText:
-  **Large-Scale**: we provide 1,500+ videos with more than 1,500,000 frame images, four times larger than the existing largest dataset for text in videos. 
-  **Open Scenario**:MOVText covers 30+ open categories with a wide selection of various scenarios, e.g., life vlog, sports news, automatic drive, cartoon, etc. Besides, caption text and scene text are separately tagged for the two different representational meanings in the video. The former represents more theme information, and the latter is the scene information. 
-  **Multilingual**:MOVText provides multilingual text annotation to promote multiple cultures live and communication.
<img src="Dataset/image/fig1.png" width="100%" class="aligncenter">

## News

## Tasks and Metrics
The proposed MOVText support four task(text detection, recognition, tracking, spotting), but mainly includes two tasks: 
-  Video Frames Detection. 
-  Video Frames Recognition. 
-  Video Text Tracking. 
-  End to End Text Spotting in Videos. 

MOTP (Multiple Object Tracking Precision)[1], MOTA (Multiple Object Tracking Accuracy) and IDF1[3,4] as the three important metrics are used to evaluate task1 (text tracking) for MMVText.
In particular, we make use of the publicly available py-motmetrics library (https://github.com/cheind/py-motmetrics) for the establishment of the evaluation metric. 

Word recognition evaluation is case-insensitive, and accent-insensitive. 
The transcription '###' or "#1" is special, as it is used to define text areas that are unreadable. During the evaluation, such areas will not be taken into account: a method will not be penalised if it does not detect these words, while a method that detects them will not get any better score.

The evluation guidance coming soon...



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
				“category”: title/caption/scene text
			},

               …

            {
				"points": [x1, y1, x2, y2, x3, y3, x4, y4],
				“tracking ID”: "#" ,
				“transcription”: "###",
				“category”: title/caption/scene text
			}
			],

“frame_2”:  
            [
			{
				"points": [x1, y1, x2, y2, x3, y3, x4, y4],
				“tracking ID”: "1" ,
				“transcription”: "###",
				“category”: title/caption/scene text
			},

               …

            {
				"points": [x1, y1, x2, y2, x3, y3, x4, y4],
				“tracking ID”: "#" ,
				“transcription”: "###",
				“category”: title/caption/scene text
			}
			],

……

}
```


### Downloads
Training data and the test set can be found from Baidu Drive(TDB) or Google Drive(TDB). (coming soon ...)

## Table Ranking

<table>
    <thead align="center">
       <tr>
           <th rowspan=2>Method</th>
		   <th colspan=5>Text Tracking Performance</th>
		   <th colspan=5>End to End Video Text Spotting</th>
           <th rowspan=2>Published at</th>
        </tr>
        <tr>
            <th>MOTA</th>
            <th>MOTP</th>
            <th>ID<sub>P</sub></th>
            <th>ID<sub>R</sub></th>
            <th>ID<sub>F1</sub></th>
            <th>MOTA</th>
            <th>MOTP</th>
            <th>ID<sub>P</sub></th>
            <th>ID<sub>R</sub></th>
            <th>ID<sub>F1</sub</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
           <td><b><a href="https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_EAST_An_Efficient_CVPR_2017_paper.html">EAST</a></b>+<b><a href="https://ieeexplore.ieee.org/abstract/document/7801919">CRNN</a></b></td>
           <td>-0.301</td>
           <td>0.275</td>
           <td>23.5</td>
           <td>22.9</td>
           <td>23.2</td>
           <td>-0.835</td>
           <td>0.173</td>
           <td>5.3%</td>
           <td>5.1%</td>
           <td>5.2%</td>
		   <td>-</td>
        </tr>

    </tbody>
</table>

## TodoList
- [x] update evaluation metric
- [ ] update data and annotation link
- [ ] update evaluation guidance
- [ ] update Baseline
- [ ] ...

## Citation


## Feedback
Suggestions and opinions of this dataset (both positive and negative) are greatly welcome. Please contact the authors by sending email to
`weijiawu@zju.edu.cn`.

## License and Copyright
The project is open source under CC-by 4.0 license (see the ``` LICENSE ``` file).

Only for research purpose usage, it is not allowed for commercial purpose usage.

while we tried to identify video that are licensed under a Creative Commons Attribution license, we make no representations or warranties regarding the license status of each video and you should verify the license for each image yourself.

## References

[1] Dendorfer, P., Rezatofighi, H., Milan, A., Shi, J., Cremers, D., Reid, I., Roth, S., Schindler, K., & Leal-Taixe, L. (2019). CVPR19 Tracking and Detection Challenge: How crowded can it get?. arXiv preprint arXiv:1906.04567.

[2] Bernardin, K. &amp; Stiefelhagen, R. Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. Image and Video Processing, 2008(1):1-10, 2008.

[3] Ristani, E., Solera, F., Zou, R., Cucchiara, R. & Tomasi, C. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. In ECCV workshop on Benchmarking Multi-Target Tracking, 2016.

[4] Li, Y., Huang, C. &amp; Nevatia, R. Learning to associate: HybridBoosted multi-target tracker for crowded scene. In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2009.



