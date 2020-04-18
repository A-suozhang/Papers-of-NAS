# Awesome-Neural-Architecture-Search

> 便于个人查找的一些NAS文章的梳理以及简要Digest
> 采用了与[Awesome-NAS](https://github.com/D-X-Y/Awesome-NAS)不同的逐模块   梳理方式，便于个人理解与速查

## Paper List

> 有编号的文章是早期比较重要的文章，Follow其的文章，会用9-A的格式来标注

|Title 📕|Source 🎓|Code 💻|Component 🔨|Property 💠|
|--|--|--|--|--|
|[1. Neural Architecture Search with Reinforcement Learning](https://arxiv.org/pdf/1611.01578)|ICLR2017(1611) *Zoph.* at Google Brain| - | Flow |NAS Flow|
|[2. Accelerating Neural Architecture Search Using Performance Prediction](https://arxiv.org/pdf/1705.10823)|ICLR2018W(1705) *Baker* at MIT| - |Evaluator|Predictor-based Evaluator|
|[3. eNAS - Efficient Architecture Search by network transformation](https://arxiv.org/abs/1707.04873)|AAAI2018(1707) *Cai* at SJTU|-|Flow/Weights-Manager|Shared Weights/Mutation from Existing Network/RL Controller|
|[4. Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/pdf/1707.07012.pdf)|CVPR2018(1707) *Zoph* Google Brain|-|Search Space|Cell-based Search Space|
|[5. HyperNet - SMASH: One-Shot Model Architecture Search through HyperNetworks](https://arxiv.org/abs/1708.05344)|ICLR2017 *Brook*|-|Weights-Manager/Evaluator|HyperNet 2 Produce SubNet's Weight|
|[5-A. Graph HyperNetwork for Neural Architecture Search](https://arxiv.org/abs/1810.05749)|ICLR2019 *Chris Zhang* Toronto|-|Weights-Manager/Evaluator|HyperNet 2 Produce SubNet's Weight|
|[6. ENAS - Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/pdf/1802.03268.pdf)|ICML2018(1802) *Pham* (at) Google Brain|-|Flow/Weight-Manager/Evaluator|Shared Weights Flow|
|[7. DARTS - Differentiable Architecture Search](https://arxiv.org/pdf/1806.09055)|ICLR2019(1806) *Liu* (at) Google Brain|-|Flow/Controller|Gradient-based Flow|
|[7-A. SNAS - Stochastic Architecture Search](https://arxiv.org/pdf/1812.09926)|ICLR2019(1812) *Xie* (at) SenseTime |-|Controller|Gradient-based Flow|
|[7-B. DARTS-nds - On Network Design Spaces for Visual Recognition](https://arxiv.org/pdf/1905.13214.pdf)|ICCV2019(1905) *IIija* (at) FAIR |-|SS|Improved NAS SS|
|[8. Hierarchical Representations for Efficient Architecture Search](https://arxiv.org/pdf/1711.00436)|ICLR2018(1711) *Liu* (at) Google Brain|-|Search Space|Hierarchical SS|
|[9. Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559)|ECCV2018(1712) *Liu* (at) Google AI|-|Controller|Predictor-based/Easy2Hard|
|[10. NAO - Neural Architecture Optimization](https://arxiv.org/abs/1808.07233)|NIPS2018(1808) *Luo* (at) MSRA|-|Evaluator|Predictor-based/Gradient-based|
|---------------------------------------------|-----------------|-----|--------------------|--------------------|
|[DetNAS: Backbone Search for Object Detection](http://arxiv.org/abs/1903.10979)|Arxiv(1903) *SunJian* at Megvii|-|Task|Shared-Weights4DetBackbone|
|[NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection](http://arxiv.org/abs/1904.07392)|Arxiv(1904) *Quo V Le* at Google Brain|-|Task/Search Space|Search for FPN|
|[EfficientDet: Scalable and Efficient Object Detection](http://arxiv.org/abs/1911.09070)|Arxiv(1911) *Quo V Le* at Google Brain|-|Task/Search Space|BiFPN+Weighted+Scalable Arch|
|---------------------------------------------|-----------------|-----|--------------------|--------------------|
|[A Survey on Neural Architecture Search](https://arxiv.org/pdf/1905.01392.pdf)|Arxiv(1905) *Martin* at IBM|-|Survey|-|
|[Accelerator-Aware Neural Network Design Using AutoML](https://arxiv.org/abs/2003.02838)|MLsys20-W Gupta|-|Hardware|NAS4Accelerator|
|[MTL-NAS: Task-Agnostic Neural Architecture Search towards General-Purpose Multi-Task Learning](https://arxiv.org/abs/2003.14058)|CVPR2020 Gao|-|Flow|NAS + MultiTasking|
|[GreedyNAS: Towards Fast One-Shot NAS with Greedy Supernet](https://arxiv.org/abs/2003.11236)|CVPR2020 You|-|Weights-Manager/Evaluator|Improvement of HyperNet|
|[How to Train Your Super-Net: An Analysis of Training Heuristics in Weight-Sharing NAS](https://arxiv.org/abs/2003.04276)|Arxiv(2003)|-|One-Shot|Analysis for Training One-Shot|
|[Disturbance-immune Weight Sharing for Neural Architecture Search](https://arxiv.org/abs/2003.13089)|Arxiv(2003) Niu|-|One-Shot|Improve Weight -Sharing|
|[NPENAS:Neural Predictor Guided Evolution for Neural Architecture Search](https://arxiv.org/abs/2003.12857)|Arxiv(2003) Wei|-|Predictor|Predictor+Evo|
|[DA-NAS:Data Adapted Pruning for Efficient Neural Architecture Search ](https://arxiv.org/abs/2003.12563)|Arxiv(2003) Dai|-|Flow|Speed up NAS flow via triming blocks|
|[MiLeNAS: Efficient Neural Architecture Search via Mixed-Level Reformulation](https://arxiv.org/abs/2003.12238)|Arxiv(2003) He|-|Gradient-based Flow|Bi-level Optimization lead to Sub-optimal|
|[Are Labels Necessary for Neural Architecture Search? ](https://arxiv.org/abs/2003.12056)|Arxiv(2003) Liu & Kaiming|-|Evaluator|Unsupervised Learning for NAS Evaluator|
|[Sampled Training and Node Inheritance for Fast Evolutionary Neural Architecture Search](https://arxiv.org/abs/2003.11613)|Arxiv(2003) Zhang|-|Controller|Faster EVO|
|[BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models](https://arxiv.org/abs/2003.11142)|Arxiv(2003) Yu|-|Shared-Weights|No Retraining Weight(like OFA)|
|[PONAS: Progressive One-shot Neural Architecture Search for Very Efficient Deployment](https://arxiv.org/abs/2003.05112)|Arxiv(2003) Huang|-|Evaluator|Progressive+One-shot|
|[Steepest Descent Neural Architecture Optimization: Escaping Local Optimum with Signed Neural Splitting](https://arxiv.org/abs/2003.10392)|Arxiv(2003) Wu|-|Evaluator|Improved NAO Flow,escape Local minima|
|[Real-time Federated Evolutionary Neural Architecture Search](https://arxiv.org/abs/2003.02793)|Arxiv(2020) Zhu|-|Flow|Evo+Federated Learning|
|[ADWPNAS: Architecture-Driven Weight Prediction for Neural Architecture Search](https://arxiv.org/abs/2003.01335)|Arxiv(2020) Zhang|-|Evaluator(HyperNet)|HyperNet Improve|
|[BS-NAS: Broadening-and-Shrinking One-Shot NAS with Searchable Numbers of Channels](https://arxiv.org/abs/2003.09821)|Arxiv(2003) Shen|-|Evaluator/Weights-Manager|Controllable Shared-Weights|
|[Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection](https://arxiv.org/abs/2003.11818)|Arxiv(2003) Guo|-|Task|HierNAS + Det|
|[DCNAS: Densely Connected Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/abs/2003.11883)|Arxiv(2003) Zhang|-|Task|NAS4Semantic Seg|
|[Probabilistic Dual Network Architecture Search on Graphs](https://arxiv.org/abs/2003.09676)|Arxiv(2003) Zhao|-|Task|NAS4GNN|
|[GAN Compression: Efficient Architectures for Interactive Conditional GAN](https://arxiv.org/abs/2003.08936)|Arxiv(2003) Li|-|Task|NAS4GAN|
|[ElixirNet: Relation-aware Network Architecture Adaptation for Medical Lesion Detection](https://arxiv.org/abs/2003.08770)|Arxiv(2003) Jiang|-|Task|NAS+Medical|
|[Lifelong Learning with Searchable Extension Units](https://arxiv.org/abs/2003.08559)|Arxiv(2003) Wang|-|Task|NAS+Continual Learning|
|[Hierarchical Neural Architecture Search for Single Image Super-Resolution](https://arxiv.org/abs/2003.04619)|Arxiv(2003) Guo|-|Task|HierNAS4SR|
|[BATS: Binary ArchitecTure Search](https://arxiv.org/abs/2003.01711)|Arxiv(2020) Bulat|-|Task|NAS4BNN|
|[NAS-Count: Counting-by-Density with Neural Architecture Search](https://arxiv.org/abs/2003.00217)|Arxiv(2020) Hu|-|Task|Counting|








## Paper Digest


```
---------------Format---------------
* 🔑 Key:         核心
* 🎓 Source:      来源
* 🌱 Motivation:  故事
* 💊 Methodology: 方法
* 📐 Exps:        实验
* 💡 Ideas:       想法
-----------------------------------
```


#### 1. [NAS with RL](https://arxiv.org/pdf/1611.01578.pdf) - Google Brain
* 🔑 Key:
  * NAS Work Flow
* 🎓 Source：ICLR2017 / Google Brain
* 🌱 Motivation:
  * 领域开端，NAS的开山,奠定了大致Workflow
  * 建立在认为NN的connectivity和structure可以被表达成一个variable-length string（可用RNN做embedding，也就是作为controller）
    * 将采样（RNN生成）出来的子架构进行训练，将Acc作为reward信号，通过Policy Gradient的方式回传去update controller（Original RNN）
* 💊 Methodology: 
  * search space
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200203201724.png)
  * RNN
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200203202126.png)
    * 将RNN看作一个树的结构
* 📐 Exps: 
  * 比cifar10上的baseline提了0.1个点，并且快了一丢丢1.05，以及其他的dataset-PennTreebank以及language modeling的task
* 💡 Ideas:
  * 在relatedwork中讲到
    * hyper-param optimization只能够做到在fixed-length的space进行模型优化，且对good initial model比较依赖
    * Bayesian方法可以寻找一个“不定长”的架构，但是不是很有意义
  * The controller in Neural Architecture Search is auto-regressive, which means it predicts hyperparameters one a time, conditioned on previous predictions. This idea is borrowed from the decoder in end-to-end sequence to sequence learning. 


#### 2. [Accelerating NAS using performance prediction](https://arxiv.org/pdf/1705.10823.pdf)

* 🔑 Key: 
  * **Predictor**        
* 🎓 Source: ICLR2018W / CMU   
* 🌱 Motivation:  
  * **提出predictor来加速evaluation**,以及EarlyStopping
  * 认为human是从training curve来观察的，本文parameterize了这个过程，训练regression model来预测acc
  * 对应的counterpart是Bayesian的一些方法
    * 目测是人工设计一些拟合learning curve的base function，然后用expensive的MCMC来拟合
    * 也有用高斯过程核函数的
* 💊 Methodology: 
  * 说白了建模的是Training Curve
    * 输出val acc，输入是configuration，在每个time step
    * 做了一个SRM（Sequential Regression model）比如RBM，RandomForest等
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200203204919.png)
* 📐 Exps:        
* 💡 Ideas:      



#### 3. [eNAS - Efficient Architecture Search by Network Transformation](https://arxiv.org/pdf/1707.04873.pdf)

* 🔑 Key:  
  * **Weight Manager** -> Shared Weights
* 🎓 Source:  AAAI2018 / SJTU
* 🌱 Motivation:  
  * **Reusing Weight(Weight Manager)不需要from scratch来训练网络**
  * metacontroller采样架构based on Network transformation 
* 💊 Methodology: 
    * 依然是Rl Controller - Policy Gradient更新，其action是指导transformation，网络的结构encoder依然是一个Bi-LSTM
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200216190254.png)
      * 有两种Actor形式，Net2Wider或者Net2Deeper
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200216190452.png)
      * Wider用一个公用的sigmoid分类器来生成反馈信息，确定是否在该时刻需要expand
      * Net2Deeper则是是否需要insert一个新的layer
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200216190710.png)
      * Deeper用一个RNN来建模是否需要在某个位置插入一个新层
      * 可以插入新层的位置是预先固定的，依据pooling层的位置把网络分成几个block
* 📐 Exps:       
* 💡 Ideas:   
    * (也可以认为是muation-based，不过指导mutation产生的不是遗传或者SA这类heuristic的方法，但是是RL-Agent)


#### 4. [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/pdf/1707.07012.pdf)
* 🔑 Key:      
  * **Cell-based Search Space** (NASNet SS)   
* 🎓 Source:     
  * CVPR2018 / Google Brain 
* 🌱 Motivation:  
  * 将SearchSpace分成多个相同Cell的堆叠
* 💊 Methodology:
  * Convolutional Blocks Repeated many times
    	* 一般分成N个Normal Cellh后面接一个Reduction Cell
  * 基于RNN的Controller
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200308185419.png)
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200308185525.png)
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200308185626.png) 
* 📐 Exps:       
* 💡 Ideas:    
	* Regularization Technique - Schedule Drop Path



#### 5. [SMASH: One-Shot Model Architecture Search through HyperNetworks](https://arxiv.org/abs/1708.05344)

* 🔑 Key:  
  * **HyperNet** -> Produce Weight
  * **One-shot NAS Framework** -> Shared Weights      
* 🎓 Source:  
  * ICLR2017 
* 🌱 Motivation: 
  * **one-shot表示从一个HyperNet中取出架构，不需要training作evaluation，采用weight sharing** 
  * HyperNet用来对给定的架构生成Weights，做到Fast Eavluation
* 💊 Methodology: 
   * HyperNet - Training an auxiliary HyperNet to generate weights
     * 加速arch selection
     * 从binary coded到optimal architecture weights的mapping，只训练output layer(?)
     * 其训练是用gradient-based的方式来做到的
* 📐 Exps:       
* 💡 Ideas:      
   * 选取arch来evluate的方式
     * Memory-bank view，将其作为binary vector
     * 是否意味着会遍历整个search space？
   * 文章中提到了说*训练一个arch开始的部分是整体acc的一个insight*
     * 一般的方法把arch的perf看作一个black box，用BO或者rs去搜索，也有early-stopping这样的策略 
   * 能够抛弃其他所有的hyper-param以及dynamic-regulariaztion的东西
     * 比如lr schedule
     * 比如DropPath之类的东西
   * [Meta-Pruning(ICCV2019)](https://arxiv.org/pdf/1903.10258)和这个有点像的
   * We hypothesize that so long as the HyperNet learns to generate reasonable weights, the validation
error of networks with generated weights will correlate with the performance when using normally
trained weights



#### 5-A. [Graph HyperNetwork for Neural Architecture Search](https://arxiv.org/pdf/1810.05749.pdf)
* 🔑 Key:  
  * 图形式的Hypernet
* 🎓 Source:  
  * ICLR 2019 && Uber & Toronto Univ.
* 🌱 Motivation: 
  * 1st to Generate All Weights
  * 多种HyperNet的方式：
    * 3D Encoding Tensor
    * LSTM Process a Sequence
* 💊 Methodology: 
  * SS
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200413183600.png)
  * GNN(生成Graph Embedding)每个Node是一个LSTM，生成一个Embedding
  * HyperNet就是一个MLP，对所有Node公用，输出大小固定
    * 对于不同大小的Weight，用堆叠Kernel或者是Channel达成
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200413183534.png)
  * *AnyTime Prediction* *Forward Backward Pass*  
    * 更新的步骤仿照网络前向以及反向的步骤
    * TimeStep 2V-1
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200413182757.png)
* 💡 Ideas:     
  * 所谓Motif的方式更为高效，搜索一个Cell，对于新的任务，就改变Cell的堆叠模式就可以了

#### 6. [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/pdf/1802.03268.pdf)

* 🔑 Key:
  * **SuperNet**
  * **One-Shot** -> Shared Weights（Current Popular Flow）     
* 🎓 Source:      
  * ICML 2018 / Google Brain
* 🌱 Motivation:
  * 构建一个Supernet，认为所有架构都是这个SuperNet的Computation Graph的一个SubGraph，各个Child Module Share Params  
* 💊 Methodology:
   * Controller在一个庞大的Compuation Graph上搜索Subgraph作为子图
    * 对于child modules Share Parameter  
  * RNN Controller
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200308232519.png)
      * 采样出链接关系
      * 以及node是什么
    * 这个controller是怎么训练的？
      * Policy Gradient？
* 📐 Exps:       
* 💡 Ideas:  



#### 7. [DARTS：Differentiable Architecture Search](https://arxiv.org/pdf/1806.09055.pdf)

* 🔑 Key:     
  * **离散搜索 -> Gradient-based**    
* 🎓 Source:   
  * ICLR 2019 / Google Brain   
* 🌱 Motivation: 
  * 将原先的离散搜索，改为Differentiable方式(Relax the Search space to be Differentiable),以提高效率！
  * 核心在于如何修改Search Space 
* 💊 Methodology: 
  * Search Space如何设计
    * 一个Cell是一个DAG，Node是一个latent representation(代表Feature Map)，每个有向的边和某种操作(Op)有关
      * 每个op以softmax来relax，取各个实际操作(Max-pool/Conv/No)发生的概率
    * 认为每个cell有两个输入一个输出，对卷积层输入来自previous 2 layer(? 2度近邻居？)
    * N=7 Node,没有strided
  * Bi-level Optimization
    * Learn Arch and Weight at the same time
    * 注意实际对alpha导数是用的，而不是单纯的 \partial{L_val}/\partrial{\alpha}
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200312222701.png)
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200312221224.png)
* 📐 Exps:       
* 💡 Ideas: 



#### 8. [Hierarchical Representations for Efficient Architecture Search](https://shimo.im/sheets/TkdXd9ptKTjDY83R/MODOC)
* 🔑 Key:     
  * **Hierarchical Search Space** 
* 🎓 Source:   
  * ICLR2018 / Google Brain   
* 🌱 Motivation:  
  * 流行的Cell-Based的架构相对general，但是有predefined的meta-arch(比如这几个Cell应该怎么堆叠之类的)不够general
* 💊 Methodology: 
  * hierarchical genetic representation
    * 模仿的是modularized design pattern
  * 采用了EA，指出最naive的random search也可以获得不错的效果
  * flat representation - 将NN作为一个DAG
    * 小的graph motif组成一个大的graph motif
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20200308204233.png)
  * Tournament Selection Search
* 📐 Exps:       
* 💡 Ideas: 

#### 9. [PNAS - Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559)
* 🔑 Key:      
  * **Progressive** -> Easy2Hard
* 🎓 Source:
  * ECCV 2018 / Google AI      
* 🌱 Motivation:  
  * Progressive (Simple 2 Complex)
* 💊 Methodology:
  * Cell-based Search Space 
  * NO RL or EA, Sequential-Model-Based Method
  * 每一步做一个局部的Heuristic Search，以前一步的Predictor来选取下一次的Predictor
* 📐 Exps:       
* 💡 Ideas:   



#### 7-A. [SNAS-Stochastic NAS](https://arxiv.org/pdf/1812.09926.pdf)

* 🔑 Key: 
  * **Gumble Softmax** in DARTS    
* 🎓 Source:      
  * ICLR2019 / SenseTime
* 🌱 Motivation:  
* 💊 Methodology: 
* 📐 Exps:       
* 💡 Ideas:  



#### 10. [NAO-Neural Architecture Optimization](https://arxiv.org/abs/1808.07233)

* 🔑 Key: 
  * Arch **Encoder-Decoder**   
* 🎓 Source:      
  * NIPS2018 / MSRA
* 🌱 Motivation:  
  * Predictor-based
  * 涉及了一个Arch的Encoder，将arch映射到一个连续空间，在该空间用Gradient优化
* 💊 Methodology:
    * Discrete -> Continuos 
      * 包含了一个encoder，将arch映射到一个连续空间，同时还搭配一个decoder
      * 还有predictor 
    * 与此类似的是DARTS，说DARTS认为最好的arch是当前weight下的argmax，而NAO直接用一个decoder映射回模型
      * 还有一支是Bayesian Optimization，作者认为GP的性能于Covariance Function的设计强相关
    * Search Space设计
      * 两步，首先决定1）which 2 previous nodes as inputs 2)确定要用什么op
    * Encoder和Decoder都是LSTM，predictor是一个mean-pooling加mlp
    * 三者Jointly Train
      * 认为predictor could work as regularization去避免encoder只对应decoder的结果，而没有正常表征
        * 这一步和传统VAE中的加noise一致
    * 认为weight-sharing和NAO是complementary的
* 📐 Exps:       
* 💡 Ideas:  
    * symmetric的design:为了保证symmetric的模型（实际上是一个模型）的embedding一致，predictor给出差不多的结果
      * 用了Augmentation（flip）来训练encoder









---


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
  * ICCV 2019 / Xuanyi Dong, Yi Yang
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


---

### [How to Train Your Super-Net: An Analysis of Training Heuristics in Weight-Sharing NAS](https://arxiv.org/abs/2003.04276)
* 🔑 Key:         
  * Analysis of One-Shot NAS SuperNet Training
* 🎓 Source:      
  * Arxiv 2003
* 🌱 Motivation:  故事
  * 对于SuperNet训练的一个Ablation
  * 需要好好调参，不应该用尽量少的epoch而应该用Subset
* 💊 Methodology: 方法
  * 用FairNAS作为平台
  * **主要重点在于训练参数**(是否有Affine，LR，WD)
  * 评价指标是超网络的准确率，超过随机搜索的概率,还有kendall-tau。
  * 最后一个是从超网络随机采样200个挑3个最好的，取他们的ground-truth的平均
* 📐 Exps:        实验
* 💡 Ideas:       想法


### [DetNAS: Backbone Search for Object Detection](http://arxiv.org/abs/1903.10979)
* 🔑 Key:   
  * nas 4 Det backbone
  * Supernet
* 🎓 Source:  
  * Megvii
* 🌱 Motivation: 
  * Det often needs imagenet pretraining & NAS requires accuracy as supervised signal
    * Imagenet-pretraining + Det finetune
  * Pre-training and finetuning are costly
    * Following One-shot, decouple the weight training and the architecture
  * Det task perf. as guide, to search for the Backbone Feature Extractor
* 💊 Methodology:
    * Steps: 
      1. SuperNet Pretrain on ImageNet and finetune on Det Task
      2. NAS on Supernet with EA
        * Path-wise(at one time, only updating samples path)
    * Finetuning BN
      * Freezing BN(Traditional) wont work for supernet(normalize couldnt acquired at different paths)
      * SyncBN replace regular BN
        * Compute BN Statistics across multiple GPUs (save memory consumption)
      * Also when EA searches arch, each BN param should be independent
        * need to re-accumulate the BN for every new arch
    * SS Design
      * Small/Big (40/20 Blocks)
        * Small used in Ablation Study
      * based on ShuffleNetV2 Block(involves channel split and shuffle operation)
        * 4 Choices:
        * x3: kernel-size [3,5,7]
        * x1: replacing right branch with Xception block(3 repeated DW 3x3 Conv)
        * 4^(40) choices for big ss
    * EA
      * Mutation + CrossOver
      * arch dont meet constraint will be removed when updating
* 📐 Exps: 
* 💡 Ideas:
  * Det's Direction
    * Architecture: FPN - Top-Down arch with lateral connection, integrating features at all scales
    * Loss: RetinaNet's Focal Loss, dealing with the class imbalance(Instability in earlier training?)
    * MetaAnchor: dynamic anchor mechanism
  * AmeobaNet - plain EA without Controller could also achieve
  * Det often uses the image classification backbone which could be sub-optimal(DetNet59 > ResNet101)
  * NAS-FPN only searches for the Feature Pyramid Network
  * EA could better handle Constraints than RL & Gradient-based 
  * [SyncBN](https://tramac.github.io/2019/04/08/SyncBN/)
    * Plain BN with DataParallel - Input distiributed to subsets and build different models on different GPU
    * (Since independent for each GPU, so batch size actually smallen)
    * Key2SyncBN: Get the global mean & var
      * Sync 1st then calc the mean & var
      * (Simplest implemention is first sync together calc mean, then send it back to calc var) - Sync twice
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master//img/20200418111031.png)
        * modify computation flow and only sync once
    * [Code](https://github.com/tamakoji/pytorch-syncbn)
  * ------------------------------------------
  * Rather Small SS
  * Simple Shared-Weights with interesting handle of BN


### [NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection](http://arxiv.org/abs/1904.07392)
* 🔑 Key:   
  * Search a FPN(Feature Pyramid Network) in a ss covering all cross-scale connections
* 🎓 Source:
  * Quoc V Le Google  
* 🌱 Motivation: 
  * Huge design space(increase exponentially)
* 💊 Methodology:
  * Following Cell-based SS(author called it as scalable architecture), main contribution is designing **search space**
    * the SS is **modular**  (Repeat the FPN N times then concat into a large net)
  * RNN RL Controller
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master//img/20200418171859.png)
  * Feature Pyramid Network
    * (following RetinaNet, use last layer in each group of feature map as *the input of the FPN*)
    * 5 Scales(C_{3,4,5,6,7}) stride of 8/16/32/64/126 pixel（6,7 purely max-pooling of 5） for Merging Cell
    * Composed of multiple merging cells  
    * Input/Output same size - can be stacked (num to stack would control the acc/flops trade-off)
    * Each merging cell gives output are appended into the candidate layers, also feeds into next merging cell
      * Finally the 5 merging cells are output 
  * Merging Cell
    * basic element of FPN
    * Merging 2 input feature map of different size
    * Each Cell has its own resolution(the output)
    * 4 Step: 
      1. choose input-layer-1 
      2. choose input-layer-2 
      3. choose output feature resolution
      4. select binary op(add or maxpool) - (scales are handles b4 the binary op)
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master//img/20200418173400.png)
  * Meshgrid Representation
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master//img/20200418173458.png)
* 📐 Exps:
  * Scalable: Stacking NAS-FPN blocks could improve acc while stacking simple block couldnt
* 💡 Ideas:
  * FPN - fuse features across different scales (cross-scale connection in ConvNets)
    * Deeper layer feature semantically strong but less resolution, are upsampled and added with lower-representation for feature with both good semantic and resolution  
    * sequentially combining 2 adjacent layer feature (with top-down/lateral connection)
    * RW
      1. [Path aggregationnetwork for instance segmentation. In CVPR, 2018.]() - add an additional bottom-up pathway
      2. [M2det: A single-shot object detector based on multi-level feature pyramid network. AAAI, 2019.]() - Multiple U-shaped Modules
      3. [Deep feature pyramid reconfiguration for object detection. In ECCV, 2018]() - Combine features at all scale + Global attention
    * Problem: manually designed and shallow(compared to backbone)
  * Any-time detection: dont necessarily need to forward all pyramid networks
    * Desired when computation effort is concern
  * ----------------------------------------------------------------------------------
  * Google's Work, really strange Hyper-param（lr-0.08/8 epochs training） (Maybe Grid-Search Again?)


### [EfficientDet: Scalable and Efficient Object Detection](http://arxiv.org/abs/1911.09070)
* 🔑 Key:   
	* Systematically NAS for Det Task
  * Combining EfficientNet Backbone + Bi-FPN + Compound Scaling
* 🎓 Source:  
	* Quo V Le Google Brain
* 🌱 Motivation: 
	* Weighted Bi-directional FPN - for multi-scale feature fusion (Better Feature Aggregation)
	* Compound Scaling method - uniformly scale the resolution/depth/width (Scalable)
* 💊 Methodology:
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master//img/20200418223126.png)
  * the "BiFPN", analogy of FPN & PANet
    * from traditional "Top-down" structure(1-way information flow)
    * Adding an extra edge when I/O is at the same level
    * remove node with only one input edge 
    * Adding Weighted Feature Fusion - like Attention
    * exponential scale up BiFPN width
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master//img/20200418220047.png)
* 📐 Exps:
* 💡 Ideas:
  * One-Stage Det: (Anchor-Free) whether have a region proposal step









## Reference

* [Awesome-NAS](https://github.com/D-X-Y/Awesome-NAS)
* [AutoML.org](https://www.automl.org/automl/literature-on-neural-architecture-search/)

 
