# Awesome-Unsupervised-Person-Re-identification[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

## KeyWords

    Unsupervised Person Reidentification,Tranfer Learning,Domain Adaptation,Clustering

## Table of Contents

- [Datasets](#Datasets)
- [Methods](#Methods)
- [Paper list](#Paper-list)
    - [Papers Updated June 16.2021](#Papers-Updated-June-16.2021)
- [Benchmarks](#Benchmarks)
  - [UDA re-ID](#UDA-re-ID)
  - [Pure re-ID](#Pure-re-ID)
- [Experience](#Experience)
- [Contributing](#Contributing)

## Datasets

- Awesome re-id dataset [[github](https://github.com/NEU-Gou/awesome-reid-dataset)]
- Market-1501 Leaderboard [[page](https://jingdongwang2017.github.io/Projects/ReID/Datasets/result_market1501.html)]
- Duke Leaderboard [[page](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/leaderboard)]
- Re-id dataset collection [[page](http://robustsystems.coe.neu.edu/sites/robustsystems.coe.neu.edu/files/systems/projectpages/reiddataset.html)]

## Methods

- [Unsupervised Domain Adaptation](#Unsupervised-Domain-Adaptation)
    - [Image-style transfer based](#Domain-style-transfer-or-Data-Augmentation)
    - [Representation learning based](#Representation-learning-based)
    - [Target domain clustering design](#Target-domain-clustering)
- [Pure re-ID](#Pure-re-ID)
    - [Handcraft feature](#Handcraft-feature)
    - [Tracklet based](#Tracklet-based)
    - [Clustering](#Clustering-based)

## Benchmarks
### UDA re-ID
- [Duke to Market](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-duke-to)
- [Market to Duke](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-market-to)
- [Market to MSMT](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-market-to-1)
- [Duke to MSMT](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-duke-to-1)

### Pure re-ID bechmarks
- Comming soon....

## Paper list
- [Papers Updated June 16.2021](#Papers-Updated-June-16.2021)
- [Unsupervised Domain Adaptation](#Unsupervised-Domain-Adaptation)
    - [Image-style transfer based](#Domain-style-transfer-or-Data-Augmentation)
    - [Representation learning based](#Representation-learning-based)
    - [Target domain clustering design](#Target-domain-clustering)
- [Pure re-ID](#Pure-re-ID)
    - [Handcraft feature](#Handcraft-feature)
    - [Tracklet based](#Tracklet-based)
    - [Clustering](#Clustering-based)
- [Other Unsupervised Learning research on Computer Vision](#Other-Unsupervised-Learning-research-on-Computer-Vision)   


### Papers Updated June 16.2021

[1] **Unsupervised Pre-Training for Person Re-Identification**(CVPR2021)[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_Unsupervised_Pre-Training_for_Person_Re-Identification_CVPR_2021_paper.pdf)]

[2] **Unsupervised Multi-Source Domain Adaptation for Person Re-Identification**(CVPR2021)[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bai_Unsupervised_Multi-Source_Domain_Adaptation_for_Person_Re-Identification_CVPR_2021_paper.pdf)]

[3] **Refining Pseudo Labels With Clustering Consensus Over Generations for Unsupervised Object Re-Identification**(CVPR2021)[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Refining_Pseudo_Labels_With_Clustering_Consensus_Over_Generations_for_Unsupervised_CVPR_2021_paper.pdf)]

[4] **Joint Noise-Tolerant Learning and Meta Camera Shift Adaptation for Unsupervised Person Re-Identification**(CVPR2021)[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Joint_Noise-Tolerant_Learning_and_Meta_Camera_Shift_Adaptation_for_Unsupervised_CVPR_2021_paper.pdf)]

[5] **Joint Generative and Contrastive Learning for Unsupervised Person Re-Identification**(CVPR2021)[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Joint_Generative_and_Contrastive_Learning_for_Unsupervised_Person_Re-Identification_CVPR_2021_paper.pdf)]

[6] **Intra-Inter Camera Similarity for Unsupervised Person Re-Identification**(CVPR2021)[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xuan_Intra-Inter_Camera_Similarity_for_Unsupervised_Person_Re-Identification_CVPR_2021_paper.pdf)]

[7] **Group-aware Label Transfer for Domain Adaptive Person Re-identification**(CVPR2021)[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Group-aware_Label_Transfer_for_Domain_Adaptive_Person_Re-identification_CVPR_2021_paper.pdf)]

[8] **Camera-Aware Proxies for Unsupervised Person Re-Identification**(AAAI2021)[[Paper](https://arxiv.org/pdf/2012.10674)]

[9] **Unsupervised Domain Adaptation for Person Re-Identification via Heterogeneous Graph
Alignment**(AAAI2021)[[Paper](https://www.aaai.org/AAAI21Papers/AAAI-4639.ZhangM.pdf)]


[10] **Exploiting Sample Uncertainty for Domain Adaptive Person Re-Identification**(AAAI2021)[[Paper](https://arxiv.org/pdf/2012.08733)]

### Unsupervised Domain Adaptation
#### Domain style transfer or Data Augmentation

[1] Li, Yu-Jhe, et al. "**Cross-dataset person re-identification via unsupervised pose disentanglement and adaptation**." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.[[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Cross-Dataset_Person_Re-Identification_via_Unsupervised_Pose_Disentanglement_and_Adaptation_ICCV_2019_paper.pdf)]

[2] Liu, Jiawei, et al. "**Adaptive transfer network for cross-domain person re-identification**." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.[[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_Transfer_Network_for_Cross-Domain_Person_Re-Identification_CVPR_2019_paper.pdf)]

[3] Zhong, Zhun, et al. "**Camstyle: A novel data augmentation method for person re-identification**." IEEE Transactions on Image Processing 28.3 (2018): 1176-1190.[[Paper](https://www.researchgate.net/profile/Zhedong-Zheng-2/publication/328158599_CamStyle_A_Novel_Data_Augmentation_Method_for_Person_Re-Identification/links/5f0d194292851c38a51cd847/CamStyle-A-Novel-Data-Augmentation-Method-for-Person-Re-Identification.pdf)]

[4] Zhong, Zhun, et al. "**Generalizing a person retrieval model hetero-and homogeneously**." Proceedings of the European Conference on Computer Vision (ECCV). 2018.[[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhun_Zhong_Generalizing_A_Person_ECCV_2018_paper.pdf)]

[5] Zhong, Zhun, et al. "**Camera style adaptation for person re-identification**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.[[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhong_Camera_Style_Adaptation_CVPR_2018_paper.pdf)]

[6] Wei, Longhui, et al. "**Person transfer gan to bridge domain gap for person re-identification**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.[[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Person_Transfer_GAN_CVPR_2018_paper.pdf)]

[7] Qian, Xuelin, et al. "**Pose-normalized image generation for person re-identification**." Proceedings of the European conference on computer vision (ECCV). 2018.[[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xuelin_Qian_Pose-Normalized_Image_Generation_ECCV_2018_paper.pdf)]

[8] Deng, Weijian, et al. "**Image-image domain adaptation with preserved self-similarity and domain-dissimilarity for person re-identification**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.[[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.pdf)]




#### Representation learning based
[2] **Domain adaptive attention model for unsupervised cross-domain person re-identiﬁcation**

[3] A novel unsupervised camera-aware domain adaptation framework for person re-identiﬁcation

[4] Jin, Xin, et al. "**Style normalization and restitution for generalizable person re-identification**." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Style_Normalization_and_Restitution_for_Generalizable_Person_Re-Identification_CVPR_2020_paper.pdf)]

[5]Zhao, Fang, et al. "**Unsupervised domain adaptation with noise resistible mutual-training for person re-identification**." European Conference on Computer Vision. Springer, Cham, 2020.[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560511.pdf)]

[6] Zhong, Zhun, et al. "**Invariance matters: Exemplar memory for domain adaptive person re-identification**." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.[[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Invariance_Matters_Exemplar_Memory_for_Domain_Adaptive_Person_Re-Identification_CVPR_2019_paper.pdf)][[Code](https://github.com/zhunzhong07/ECN?utm_source=catalyzex.com)]

[7] Bak, Slawomir, Peter Carr, and Jean-Francois Lalonde. "**Domain adaptation through synthesis for unsupervised person re-identification**." Proceedings of the European Conference on Computer Vision (ECCV). 2018.[[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper.pdf)]

[8] Lin, Shan, et al. "**Multi-task mid-level feature alignment network for unsupervised cross-dataset person re-identification**." arXiv preprint arXiv:1807.01440 (2018).[[Paper](https://arxiv.org/pdf/1807.01440)]

[9] Wang, Jingya, et al. "**Transferable joint attribute-identity deep learning for unsupervised person re-identification**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.[[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Transferable_Joint_Attribute-Identity_CVPR_2018_paper.pdf)]

[10] Li, Yu-Jhe, et al. "**Adaptation and re-identification network: An unsupervised deep transfer learning approach to person re-identification**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.[[Paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Li_Adaptation_and_Re-Identification_CVPR_2018_paper.pdf)]

[11] Lv, Jianming, et al. "**Unsupervised cross-dataset person re-identification by transfer learning of spatial-temporal patterns**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.[[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lv_Unsupervised_Cross-Dataset_Person_CVPR_2018_paper.pdf)][[Code](https://github.com/ahangchen/TFusion)]

[12] Li, Yu-Jhe, et al. "**Adaptation and re-identification network: An unsupervised deep transfer learning approach to person re-identification**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.[[Paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Li_Adaptation_and_Re-Identification_CVPR_2018_paper.pdf)]

[13] Bak, Slawomir, Peter Carr, and Jean-Francois Lalonde. "**Domain adaptation through synthesis for unsupervised person re-identification**." Proceedings of the European Conference on Computer Vision (ECCV). 2018.[[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper.pdf)]

[14] Li, Yu-Jhe, et al. "**Adaptation and re-identification network: An unsupervised deep transfer learning approach to person re-identification**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.[[Paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Li_Adaptation_and_Re-Identification_CVPR_2018_paper.pdf)]

[15] Geng, Mengyue, et al. "**Deep transfer learning for person re-identification**." arXiv preprint arXiv:1611.05244 (2016).[[Paper](https://arxiv.org/pdf/1611.05244)]

[16] Peng, Peixi, et al. "**Unsupervised cross-dataset transfer learning for person re-identification**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.[[Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Peng_Unsupervised_Cross-Dataset_Transfer_CVPR_2016_paper.pdf)]

[17] Ma, Andy J., et al. "**Cross-domain person reidentification using domain adaptation ranking svms**." IEEE transactions on image processing 24.5 (2015): 1599-1613.[[Paper](https://ieeexplore.ieee.org/iel7/83/4358840/07018030.pdf)]

[18] Chen, Yanbei, Xiatian Zhu, and Shaogang Gong. "**Instance-guided context rendering for cross-domain person re-identification**." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Instance-Guided_Context_Rendering_for_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.pdf)]

[19] Huang, Houjing, et al. "**Eanet: Enhancing alignment for cross-domain person re-identification**." arXiv preprint arXiv:1812.11369 (2018).[[Paper](https://arxiv.org/pdf/1812.11369)][[Code](https://github.com/huanghoujing/EANet)]

[20] Zhang, Xinyu, et al. "**Self-training with progressive augmentation for unsupervised cross-domain person re-identification**." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Self-Training_With_Progressive_Augmentation_for_Unsupervised_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.pdf)]

[21] Wu, Jinlin, et al. "**Unsupervised graph association for person re-identification**." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Unsupervised_Graph_Association_for_Person_Re-Identification_ICCV_2019_paper.pdf)][[Code](https://github.com/yichuan9527/Unsupervised-Graph-Association-for-Person-Re-identification)]

#### Target domain clustering

### Pure re-ID

#### Clustering-based

Yunpeng Zhai, Shijian Lu, Qixiang Ye, Xuebo Shan, Jie Chen, Rongrong Ji, and Yonghong Tian. **Ad-cluster: Augmented discriminative clustering for domain adaptive person re-identification**. In CVPR, 2020. 1, 3, 8 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhai_AD-Cluster_Augmented_Discriminative_Clustering_for_Domain_Adaptive_Person_Re-Identification_CVPR_2020_paper.pdf)]

[25] Ge, Yixiao, Dapeng Chen, and Hongsheng Li. "**Mutual mean-teaching: Pseudo label refinery for unsupervised domain adaptation on person re-identification**." arXiv preprint arXiv:2001.01526 (2020).[[Paper](https://arxiv.org/pdf/2001.01526)]

[26] Wang, Dongkai, and Shiliang Zhang. "**Unsupervised person re-identification via multi-label classification**." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.[[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf)]

[27] Lin, Yutian, et al. "**Unsupervised person re-identification via softened similarity learning**." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Unsupervised_Person_Re-Identification_via_Softened_Similarity_Learning_CVPR_2020_paper.pdf)][[Code](https://github.com/ryanaleksander/softened-similarity-learning)]

[28] Wu, Jinlin, et al. "**Clustering and dynamic sampling based unsupervised domain adaptation for person re-identification**." 2019 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2019.[[Paper](http://www.cbsr.ia.ac.cn/users/zlei/papers/JLWU-ICME-2019.pdf)]

[29] Lin, Yutian, et al. "**A bottom-up clustering approach to unsupervised person re-identification**." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. No. 01. 2019.[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/4898/4771)]

[30] Ding, Guodong, et al. "**Dispersion based Clustering for Unsupervised Person Re-identification**." BMVC. 2019.[[Paper](https://guodongding.cn/papers/ding2019dispersion.pdf)]

[31] Fan, Hehe, et al. "**Unsupervised person re-identification: Clustering and fine-tuning.**" ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 14.4 (2018): 1-18.[[Paper](https://arxiv.org/pdf/1705.10444)]

[32] Ye, Mang, et al. "**Dynamic label graph matching for unsupervised video re-identification.**" Proceedings of the IEEE international conference on computer vision. 2017.[[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ye_Dynamic_Label_Graph_ICCV_2017_paper.pdf)]

[33] Yu, Hong-Xing, Ancong Wu, and Wei-Shi Zheng. "**Cross-view asymmetric metric learning for unsupervised person re-identification**." Proceedings of the IEEE international conference on computer vision. 2017.[[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yu_Cross-View_Asymmetric_Metric_ICCV_2017_paper.pdf)]

[34] Yang, Fengxiang, et al. "**Asymmetric co-teaching for unsupervised cross-domain person re-identification**." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 07. 2020.[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/6950/6804)]

[35] Fu, Yang, et al. "**Self-similarity grouping: A simple unsupervised cross domain adaptation approach for person re-identification**." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.[[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_Self-Similarity_Grouping_A_Simple_Unsupervised_Cross_Domain_Adaptation_Approach_for_ICCV_2019_paper.pdf)]

[36] Lin, Shan, et al. "**Multi-task mid-level feature alignment network for unsupervised cross-dataset person re-identification**." arXiv preprint arXiv:1807.01440 (2018).[[Paper](https://arxiv.org/pdf/1807.01440)]

[37] Jin, Xin, et al. "**Global distance-distributions separation for unsupervised person re-identification**." European Conference on Computer Vision. Springer, Cham, 2020.[[Paper](https://arxiv.org/pdf/2006.00752)]

[38] Li, Jianing, and Shiliang Zhang. "**Joint Visual and Temporal Consistency for Unsupervised Domain Adaptive Person Re-Identification**." European Conference on Computer Vision. Springer, Cham, 2020.[[Paper](https://arxiv.org/pdf/2007.10854)]

#### Tracklet based

[39] Wu, Guile, Xiatian Zhu, and Shaogang Gong. "**Tracklet self-supervised learning for unsupervised person re-identification**." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 07. 2020.[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6921/6775)]

[40] Li, Minxian, Xiatian Zhu, and Shaogang Gong. "**Unsupervised tracklet person re-identification**." IEEE transactions on pattern analysis and machine intelligence 42.7 (2019): 1770-1782.[[Paper](https://arxiv.org/pdf/1903.00535)]

[41] Li, Minxian, Xiatian Zhu, and Shaogang Gong. "**Unsupervised person re-identification by deep learning tracklet association**." Proceedings of the European conference on computer vision (ECCV). 2018.[[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Minxian_Li_Unsupervised_Person_Re-identification_ECCV_2018_paper.pdf)]

[42] Ye, Mang, Xiangyuan Lan, and Pong C. Yuen. "**Robust anchor embedding for unsupervised video person re-identification in the wild**." Proceedings of the European Conference on Computer Vision (ECCV). 2018.[[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Mang_YE_Robust_Anchor_Embedding_ECCV_2018_paper.pdf)]

[43] Ma, Xiaolong, et al. "**Person re-identification by unsupervised video matching**." Pattern Recognition 65 (2017): 197-210.[[Paper](https://arxiv.org/pdf/1611.08512)]

[44] Liu, Zimo, Dong Wang, and Huchuan Lu. "**Stepwise metric promotion for unsupervised video person re-identification**." Proceedings of the IEEE international conference on computer vision. 2017.[[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Stepwise_Metric_Promotion_ICCV_2017_paper.pdf)]

[45]	Xie, Qiaokang, et al. "**Progressive Unsupervised Person Re-identification by Tracklet Association with Spatio-Temporal Regularization**." IEEE Transactions on Multimedia

#### Handcraft feature

[1] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian.**Scalable person re-identiﬁcation: A benchmark**. In ICCV, 2015.
UMDL（2016）

[2] S. Liao, Y. Hu, X. Zhu, and S. Z. Li, “**Person reidentiﬁcation by local maximal occurrence representation and metric learning**.” in CVPR, 2015, pp. 21972206.

[3] Ma, Bingpeng, Yu Su, and Frédéric Jurie. "**Bicov: a novel image representation for person re-identification and face verification**." British Machive Vision Conference. 2012.[[Paper](https://hal.archives-ouvertes.fr/hal-00806112/file/12_bmvc-person-reid.pdf)]

[4] G. Lisanti, I. Masi, A. D. Bagdanov, and A. Del Bimbo, “**Person reidentiﬁcation by iterative re-weighted sparse ranking**,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 37, no. 8, pp. 1629–1642, 2015.     ？

[5] E. Kodirov, T. Xiang, Z. Fu, and S. Gong, “**Person re-identiﬁcation by unsupervised l 1 graph learning**,” in Proc. Eur. Conf. Comput. Vis., 2016, pp. 178–195.   ？

### Semi-supervised Learning or Few-shot Learning
[45] Li, Jiawei, Andy J. Ma, and Pong C. Yuen. "**Semi-supervised region metric learning for person re-identification**." International Journal of Computer Vision 126.8 (2018): 855-874.

[46] Wu, Yu, et al. "**Exploit the unknown gradually: One-shot video-based person re-identification by stepwise learning**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.[[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Exploit_the_Unknown_CVPR_2018_paper.pdf)]

[47] Su, Chi, et al. "**Deep attributes driven multi-camera person re-identification**." European conference on computer vision. Springer, Cham, 2016.[[Paper](https://arxiv.org/pdf/1605.03259)]


### Other methods like Metric Learning，Dictionary Learning and Salience Learning

[51] Khan, Furqan M., and Francois Bremond. "**Unsupervised data association for metric learning in the context of multi-shot person re-identification**." 2016 13th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS). IEEE, 2016.[[Paper](http://www-sop.inria.fr/members/Francois.Bremond/Postscript/FurqanAVSS2016.pdf)]

[52] Wang, Hanxiao, et al. "**Towards unsupervised open-set person re-identification**." 2016 IEEE International Conference on Image Processing (ICIP). IEEE, 2016.[[Paper](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/18790/Gong%20Towards%20unsupervised%20open-set%202016%20Accepted.pdf?sequence=1&isAllowed=y)]

[53] Kodirov, Elyor, Tao Xiang, and Shaogang Gong. "**Dictionary learning with iterative laplacian regularisation for unsupervised person re-identification**." BMVC. Vol. 3. 2015.[[Paper](http://www.bmva.org/bmvc/2015/papers/paper044/paper044.pdf)]

[54] Wang, Hanxiao, Shaogang Gong, and Tao Xiang. "**Unsupervised learning of generative topic saliency for person re-identification**." (2014).[[Paper](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/6442/GONGUnsupervisedLearning2014.pdf?sequence=2)]


[56] Liu, Xiao, et al. "**Semi-supervised coupled dictionary learning for person re-identification**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014.[[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Liu_Semi-Supervised_Coupled_Dictionary_2014_CVPR_paper.pdf)]

[57] Zhao, Rui, Wanli Ouyang, and Xiaogang Wang. "**Unsupervised salience learning for person re-identification**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2013.[[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Zhao_Unsupervised_Salience_Learning_2013_CVPR_paper.pdf)]

[58] P. Peng, T. Xiang, Y. Wang, P. Massimiliano, S. Gong, T. Huang, and Y. Tian, “**Unsupervised cross-dataset transfer learning for person re-identiﬁcation**,” in CVPR, 2016, pp. 1306–1315.


### Other Unsupervised Learning research on Computer Vision

[1] B. Fernando, A. Habrard, M. Sebban, and T. Tuytelaars. **Unsupervised visual domain adaptation using subspace alignment**. In ICCV, 2013.

[2] B. Gong, Y. Shi, F. Sha, and K. Grauman. **Geodesic ﬂow kernel for unsupervised domain adaptation**. In CVPR, 2012.

[3] R. Gopalan, R. Li, and R. Chellappa. **Domain adaptation for object recognition: An unsupervised approach**. In Computer Vision (ICCV), 2011 IEEE International Conference on, pages 999–1006, Nov 2011.

[4] Q. Qiu, J. Ni, and R. Chellappa. **Dictionary-based domain adaptation methods for the re-identiﬁcation of faces**. In Person Re-Identiﬁcation, pages 269–285. Springer, 2014.

[5] Zheng, Zhedong, Liang Zheng, and Yi Yang. "**Unlabeled samples generated by gan improve the person re-identification baseline in vitro**." Proceedings of the IEEE International Conference on Computer Vision. 2017.[[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Unlabeled_Samples_Generated_ICCV_2017_paper.pdf)]

[6] Hoffman, Judy, et al. "**Cycada: Cycle-consistent adversarial domain adaptation." International conference on machine learning**. PMLR, 2018.[[Paper](http://proceedings.mlr.press/v80/hoffman18a/hoffman18a.pdf)][[Code](https://github.com/jhoffman/cycada_release)]

[7] Dong, Xuanyi, et al. "**Style aggregated network for facial landmark detection**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.[[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Style_Aggregated_Network_CVPR_2018_paper.pdf)][[Code](https://github.com/D-X-Y/landmark-detection)]

[8] Zhao, Jian, et al. "**3D-Aided Deep Pose-Invariant Face Recognition**." IJCAI. Vol. 2. No. 3. 2018.[[Paper](https://www.researchgate.net/profile/Jian_Zhao68/publication/324557770_3D-Aided_Deep_Pose-Invariant_Face_Recognition/links/5ade859ba6fdcc29358d88ce/3D-Aided-Deep-Pose-Invariant-Face-Recognition.pdf)]

[9] Isola, Phillip, et al. "**Image-to-image translation with conditional adversarial networks**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.[[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)][[Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix?utm_source=catalyzex.com)]

[10] Bousmalis, Konstantinos, et al. "**Unsupervised pixel-level domain adaptation with generative adversarial networks**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.[[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bousmalis_Unsupervised_Pixel-Level_Domain_CVPR_2017_paper.pdf)][[Code](https://github.com/marload/GANs-TensorFlow2)]

[11] Zhang, Richard, Phillip Isola, and Alexei A. Efros. "**Split-brain autoencoders: Unsupervised learning by cross-channel prediction**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.[[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Split-Brain_Autoencoders_Unsupervised_CVPR_2017_paper.pdf)][[Code](https://github.com/ysharma1126/Split-Brain-Autoencoder)]

[12] Bojanowski, Piotr, and Armand Joulin. "**Unsupervised learning by predicting noise**." International Conference on Machine Learning. PMLR, 2017.[[Paper](http://proceedings.mlr.press/v70/bojanowski17a/bojanowski17a.pdf)][[Code](https://github.com/facebookresearch/noise-as-targets)]

[13] Chang, Jianlong, et al. "**Deep adaptive image clustering**." Proceedings of the IEEE international conference on computer vision. 2017.[[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chang_Deep_Adaptive_Image_ICCV_2017_paper.pdf)][[Code](https://github.com/vector-1127/DAC)]

[14] Zhu, Jun-Yan, et al. "**Unpaired image-to-image translation using cycle-consistent adversarial networks**." Proceedings of the IEEE international conference on computer vision. 2017.[[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)][[Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]

[15] Sun, Yifan, et al. "**Svdnet for pedestrian retrieval**." Proceedings of the IEEE International Conference on Computer Vision. 2017.[[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Sun_SVDNet_for_Pedestrian_ICCV_2017_paper.pdf)]

[16] Lee, Hsin-Ying, et al. "**Unsupervised representation learning by sorting sequences**." Proceedings of the IEEE International Conference on Computer Vision. 2017.[[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Unsupervised_Representation_Learning_ICCV_2017_paper.pdf)][[Code](https://github.com/HsinYingLee/OPN)]

[17] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "**Image style transfer using convolutional neural networks**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.[[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)]

[18] Bousmalis, Konstantinos, et al. "**Domain separation networks**." arXiv preprint arXiv:1608.06019 (2016).[[Paper](https://arxiv.org/pdf/1608.06019)]

[19] Zhao, Rui, Wanli Oyang, and Xiaogang Wang. "**Person re-identification by saliency learning**." IEEE transactions on pattern analysis and machine intelligence 39.2 (2016): 356-370.[[Paper](https://arxiv.org/pdf/1412.1908)]

[20] Taigman, Yaniv, Adam Polyak, and Lior Wolf. "**Unsupervised cross-domain image generation**." arXiv preprint arXiv:1611.02200 (2016).[[Paper](https://arxiv.org/pdf/1611.02200.pdf?source=post_page---------------------------)][[Code](https://github.com/kaonashi-tyc/zi2zi)]

[21] Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "**Perceptual losses for real-time style transfer and super-resolution**." European conference on computer vision. Springer, Cham, 2016.[[Paper](https://arxiv.org/pdf/1603.08155.pdf%7C)][[Code](https://github.com/DmitryUlyanov/texture_nets)]

[22] Sun, Baochen, Jiashi Feng, and Kate Saenko. "**Return of frustratingly easy domain adaptation**." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 30. No. 1. 2016.[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/10306/10165)]

[23] Sun, Baochen, and Kate Saenko. "**Deep coral: Correlation alignment for deep domain adaptation**." European conference on computer vision. Springer, Cham, 2016.[[Paper](https://arxiv.org/pdf/1607.01719)][[Code](https://github.com/domainadaptation/salad)]
Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." arXiv preprint arXiv:1511.06434 (2015).[[Paper](https://arxiv.org/pdf/1511.06434.pdf%C3)][[Code](https://github.com/mattya/chainer-DCGAN)]

[24] Bojanowski, Piotr, et al. "**Weakly supervised action labeling in videos under ordering constraints**." European Conference on Computer Vision. Springer, Cham, 2014.[[Paper](https://link.springer.com/content/pdf/10.1007/978-3-319-10602-1_41.pdf)]

[25] Goodfellow I J, Pouget-Abadie J, Mirza M, et al. "**Generative adversarial nets**." Proceedings of the 27th International Conference on Neural Information Processing Systems. Cambridge: MIT Press, 2014:2672-2680.

[26] Lisanti, Giuseppe, et al. "**Person re-identification by iterative re-weighted sparse ranking**." IEEE transactions on pattern analysis and machine intelligence 37.8 (2014): 1629-1642.[[Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.725.7570&rep=rep1&type=pdf)]

[2] Tzeng, Eric, et al. "**Deep domain confusion: Maximizing for domain invariance**." arXiv preprint arXiv:1412.3474 (2014).[[Paper](https://arxiv.org/pdf/1412.3474)][[Code](https://github.com/erlendd/ddan)]

[73] Fernando, Basura, et al. "**Unsupervised visual domain adaptation using subspace alignment**." Proceedings of the IEEE international conference on computer vision. 2013.[[Paper](http://openaccess.thecvf.com/content_iccv_2013/papers/Fernando_Unsupervised_Visual_Domain_2013_ICCV_paper.pdf)]

[74] Gong, Boqing, et al. "**Geodesic flow kernel for unsupervised domain adaptation**." 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012.[[Paper](https://www.cs.utexas.edu/users/grauman/papers/subspace-cvpr2012.pdf)]

[75] Gong, Boqing, et al. "**Geodesic flow kernel for unsupervised domain adaptation**." 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012.[[Paper](https://www.cs.utexas.edu/users/grauman/papers/subspace-cvpr2012.pdf)]

[77] Ma, Bingpeng, Yu Su, and Frédéric Jurie. "**Local descriptors encoded by fisher vectors for person re-identification**." European conference on computer vision. Springer, Berlin, Heidelberg, 2012.[[Paper](https://link.springer.com/content/pdf/10.1007/978-3-642-33863-2_41.pdf)]

[78] Farenzena, Michela, et al. "**Person re-identification by symmetry-driven accumulation of local features**." 2010 IEEE computer society conference on computer vision and pattern recognition. IEEE, 2010.[[Paper](http://profs.scienze.univr.it/~cristanm/Publications_files/CVPR2010_Cristani.pdf)]

### ？？ 
Kodirov E, Xiang T, Fu ZY, Gong SG. **Person re-identification by unsupervised l2 graph learning**. European conference on computer vision, Springer, Cham, 2016: 178-195.

[40] Kodirov E, Xiang T, Gong SG. **Dictionary learning with iterative laplacian regularisation for unsupervised person re-identification**. BMVC, 2015, 3: 8.

[41] Zhao R, Ouyang WL, Wang XG. **Unsupervised salience learning for person re-identification**. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2013: 3586-3593.

[42] Yang Y, Wen LY, Lyu SW, Li SZ. **Unsupervised learning of multi-level descriptors for person re-identification**. Thirty-First AAAI Conference on Artificial Intelligence, 2017.

## Experience

Comming soon...

## Contributing

Please help contribute this list by contacting [me](iminliu@yeah.net) or add [pull request](https://github.com/Yimin-Liu/Awesome-Cross-Domain-Person-Re-identification/pulls)

Markdown format:
```markdown
- References. 
  [[pdf]](link) 
  [[code]](link)
```