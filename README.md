# Awesome-Neural-Architecture-Search

> 便于个人查找的一些NAS文章的梳理以及简要Digest
> 采用了与[Awesome-NAS](https://github.com/D-X-Y/Awesome-NAS)不同的逐模块   梳理方式，便于个人理解与速查

## Genre


## Paper Digest


#### [Overcoming Multi-Model Forgetting in One-Shot NAS with DiversityMaximization](https://shiruipan.github.io/publication/cvpr-2020-zhang/)
* 🔑 Key:
  * 解决传统的Shared-Weights方法在优化新的架构的时候老的架构精度会下降(Catastrophic Forgetting)的问题
* 🎓 Source：
  * CVPR 2020
* 🌱 Motivation: 
    * 传统的OneShot方式认为Jointly Optimized Supernet Weights是最优的
    * 但是sequentially train archs with partially-shared weights会导致Catastrophic Forgetting
    * 文章核心把One-ShotNAS看成一个Continual Learning的问题(Constrained Optimzation,learning of current arch should not degrade previous much)
* 💊 Methodology:
    * NSAS(Search-based Architecture Selection) Loss Function 
    * Enforce the architectures inheriting weights from the supernet in current step perform better than last step
    * 如果累计的话要求会太高了，所以不是限定全部的Previous Arch，而是选择其中的一个Subset(对于如何选定这个Subset是假定Subset中的Arch要有Diversity-找到最大Diversity的过程就是所为的Novelty Search)
    * 实现约束的方式是加一个Soft Regularization

#### [One-Shot Neural Architecture Search via Self-Evaluated Template Network](https://arxiv.org/abs/1910.05733)
* 🔑 Key:
  * 传统的Evluation慢，Shared Weights的方式选取去Evaluate的组件的时候是Random的，不够Instructive
* 🎓 Source：
  * ICCV 2019
* 🌱 Motivation: 
  * 提出了一个SETN(Self Evaluated Template Network)
    * 一个Evaluator去预测有更低Valid Loss的架构(类似一个Predictor)
    * 一个模板Template网络去Shared Params,包含了所有的Candidate
* 💊 Methodology:
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200402210522.png)
    * 看上去就是对Shared-weights加了一个Predictor作为Controller，去从一个所谓的Template网络中采样出子架构，Controller来决定怎么采(而不是随机采样)
  * N个Cell
    * 每个Cell中B个Block
    * 每个Block可能是4元组
  * Candidate Network 
    * Contain All candidate CNN in search space
    * train stochsticly - uniformly sample 1 candidate and only optimize its params
      * Optimize each with equal possibility
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200402212418.png)
      * I是input是纯随机Sample，F是Function，其中的order指的是再一个集合O中采样，其中f1的index一定要小于f2
    * Evaluator：
      * Encode one CNN candidate as a set of quadruples
      * 从categorical distribution sample出一个choice，用softmax normalized value作为vector值
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200402213049.png)
* 💡 Ideas:
    * 指出有些Shared-Weights的方法会shared parameter with a learnable distribution of archs
      * 作者认为这样会有bias，因为相对lightweight的model会更快收敛，learnable distribution will bias to these model
      * “the Matthew effect” to refer that some quickly-converged candidates will get more chances  to be further optimized in some NAS algorithms


#### [GATES]()
* 🔑 Key: 
  * NAS中的**Predictor**问题，提供一个更好的Encoder
* 🎓 Source
  * Arxiv & THU EE
* 🌱 Motivation: 
  * Current Encoder model topological information implicitly
  * 原本的gcn之类的encoder方式edges stand for notion of affinity, feature on node
  * View NN as Data-Processing Graph
* 💊 Methodology:
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200404200848.png)
  * 认为优点在于能够直接Handle Topological isomorphism,以及对Information Processing的建模
* 📐 Exps: 
  * Nb101/201上KD，NatK，PatK
* 💡 Ideas:
  * 相关文章NAO，用一个Encoder-Decoder来做
  * In the Kendall's Tau measure, all discordant pairs are treated equally
  * 其他指标(因为kendall tau其实考虑了很多poor arch的相对关系，对于NAS来说用处不大)
    * NatK： predict给出的K个中最好的实际rank
    * PatK： Predictor TopK在Gt TopK中的比例
  * Nb101 - 432k archs - Op on Node
  * Nb201 - 15625 - OP on Edge














## Reference

* [Awesome-NAS](https://github.com/D-X-Y/Awesome-NAS)

 
