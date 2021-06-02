
The model was developed for the [Cornell Birdcall Identification](https://www.kaggle.com/c/birdsong-recognition) kaggle competition,
and it is able to recognize 262 North American species.
The architecture is slightly modified version of the [CNN14](https://github.com/qiuqiangkong/audioset_tagging_cnn) pretrained on audioset [1]. 

The training data consists tens of thousands short recordings of individual bird calls generously uploaded by users of [xeno-canto.org](https://www.xeno-canto.org/).
As proposed in [2] additional [noises](http://dcase.community/challenge2018/task-bird-audio-detection) [3] and [animal sounds](https://www.tierstimmenarchiv.de/) were used as augmentation for more robust predictions.
[Ebird Basic Dataset (EBD)](https://ebird.org/science/download-ebird-data-products) was used to upsample more common birds and to generate bird distribution maps.


### References
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley.
*PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition.*
arXiv preprint arXiv:1912.10211 (2019).

[2] Mario Lasseck. *Bird Species Identification in Soundscapes.* CEUR Workshop Proceedings, 2019.

[3] D. Stowell, Y. Stylianou, M. Wood, H. Pamuła, and H. Glotin.
*Automatic acoustic detection of birds through deep learning: the first bird audio detection challenge.*
Methods in Ecology and Evolution, 2018. URL: https://arxiv.org/abs/1807.05812, arXiv:1807.05812.

