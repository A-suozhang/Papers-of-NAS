# Papers-of-Neural-Architecture-Search

> Some nas paper for self-reference
> **paper digest at [notes.md](./notes.md)**
> Different from [Awesome-NAS](https://github.com/D-X-Y/Awesome-NAS), arrange papers according to components

## Paper List

> Papers with number idx is earlier famous paper, papers to follow are denoted like 9-A

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
|[8. Hierarchical Representations for Efficient Architecture Search](https://arxiv.org/pdf/1711.00436)|ICLR2018(1711) *Liu* (at) Google Brain|-|Search Space|Hierarchical SS|
|[9. Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559)|ECCV2018(1712) *Liu* (at) Google AI|-|Controller|Predictor-based/Easy2Hard|
|[10. NAO - Neural Architecture Optimization](https://arxiv.org/abs/1808.07233)|NIPS2018(1808) *Luo* (at) MSRA|-|Evaluator|Predictor-based/Gradient-based
|[11. ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332)|ICLR2019 MIT|-|TraniningTech|Gradient-based|

### Weight-Sharing Improvement

|Title 📕|Source 🎓|Code 💻|Component 🔨|Property 💠|
|--|--|--|--|--|
|[How to Train Your Super-Net: An Analysis of Training Heuristics in Weight-Sharing NAS](https://arxiv.org/abs/2003.04276)]|Arxiv|-|Analysis|Weight-Sharing|
|[EcoNAS: Finding Proxies for Economical Neural Architecture Search](https://arxiv.org/abs/2001.01233)|Arxiv NYU&SenseTime|-|Analysis|Weight-Sharing|
|[Improved one-shot NAS by suppressing posteriror fading](http://xxx.itp.ac.cn/pdf/1910.02543v1)|Arxiv|-|Trick|Weight-sharing|


### Darts Improvement

|Title 📕|Source 🎓|Code 💻|Component 🔨|Property 💠|
|--|--|--|--|--|
|[7-A. SNAS - Stochastic Architecture Search](https://arxiv.org/pdf/1812.09926)|ICLR2019(1812) *Xie* (at) SenseTime |-|Controller|Gradient-based Flow|
|[7-B. DARTS-nds - On Network Design Spaces for Visual Recognition](https://arxiv.org/pdf/1905.13214.pdf)|ICCV2019(1905) *IIija* (at) FAIR |-|SS|Improved NAS SS|
|[7-C. P-Darts:Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](http://arxiv.org/abs/1904.12760)|CVPR2020 Huawei & Tongji|-|Evaluator|Gradient-based Flow|
|[7-D. PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search](http://arxiv.org/abs/1907.05737)|ICLR2020-Huawei|-|Evaluator/Controller|Gradient-based flow|
|[7-E. GOLD-NAS: Gradual, One-Level, Differentiable](http://arxiv.org/abs/2007.03331)|Arxiv Huawei|-|SS|Gradient-based flow|
|[7-F. DARTS+: Improved Differentiable Architecture Search with Early Stopping](http://arxiv.org/abs/1909.06035)]|Huawei|-|TrainingTech|Gradient-based flow|
|[7-G. Stabilizing DARTS with Amended Gradient Estimation on Architectural Parameters](http://arxiv.org/abs/1910.11831)]|ICML2020 Huawei|-|TrainingTech|Gradient-based flow|
|[7-H. Fair DARTS: Eliminating Unfair Advantages in Differentiable Architecture Search](https://arxiv.org/abs/1911.12126)]|ECCV2020 Xiaomi|-|Evaluator|Gradient-based flow|
|[7-I. Efficient Neural Architecture Search via Proximal Iterations](https://arxiv.org/abs/1905.13577)]|AAAI2020 4 Paragdim/PKU|-|TrainingTech|Gradient-based flow|
|[7-J. MergeNAS: Merge Operations into One for Differentiable Architecture Search](https://www.ijcai.org/Proceedings/2020/0424.pdf)|IJCAI2020 - SJTU/IBM|-|TrainingTech|Gradient-based Flow|


### Hardware Related

|Title 📕|Source 🎓|Code 💻|Component 🔨|Property 💠|
|--|--|--|--|--|
|[TF-NAS: Rethinking Three Search Freedoms of Latency-Constrained Differentiable Neural Architecture Search](http://arxiv.org/abs/2008.05314)|ECCV2020 - CRIPAC|-|Flow|Gradient-based flow|
|[APQ: Joint Search for Network Architecture, Pruning and Quantization Policy](http://arxiv.org/abs/2006.08509)| CVPR2020 - MIT|-|Flow|Gradient-search|
|[Beyond Network Pruning: a Joint Search-and-Training Approach](http://see.xidian.edu.cn/faculty/wsdong/Papers/Conference/ijcai20.pdf)|IJCAI2020 - Xidian|-|Flow|RL|
|[Densely Connected Search Space for More Flexible Neural Architecture Search](http://arxiv.org/abs/1906.09607)|CVPR2020|-|SS|Gradient-based|
|[Shape Adaptor: A Learnable Resizing Module](https://arxiv.org/abs/2008.00892)|ECCV2020|-|NewElement in SS|Gradient-based|
|[Angle-based Search Space Shrinking for Neural Architecture Search](https://arxiv.org/abs/2004.13431)|ECCV2020 Megvll|-|Evaluator|Gradient-based|

### Det+NAS

|Title 📕|Source 🎓|Code 💻|Component 🔨|Property 💠|
|--|--|--|--|--|
|[DetNAS: Backbone Search for Object Detection](http://arxiv.org/abs/1903.10979)|Arxiv(1903) *SunJian* at Megvii|-|Task|Shared-Weights4DetBackbone|
|[NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection](http://arxiv.org/abs/1904.07392)|Arxiv(1904) *Quo V Le* at Google Brain|-|Task/Search Space|Search for FPN|
|[EfficientDet: Scalable and Efficient Object Detection](http://arxiv.org/abs/1911.09070)|Arxiv(1911) *Quo V Le* at Google Brain|-|Task/Search Space|BiFPN+Weighted+Scalable Arch|


### Binary+NAS

|Title 📕|Source 🎓|Code 💻|Component 🔨|Property 💠|
|--|--|--|--|--|
|[Binarizing MobileNet via Evolution-based Searching](http://arxiv.org/abs/2005.06305)|Arxiv(2005)|-|Task|Evo Search for group-conv MobileBlock|
|[Searching for Accurate Binary Neural Architectures](http://arxiv.org/abs/1909.07378)|ICCVW19(1909) Huawei Noah|-|Task|Evo Search width for MobileBlock|
|[Learning Architectures for Binary Networks](http://arxiv.org/abs/2002.06963)|ECCV2020(2002) GIST|-|Task|Darts+Binary|
|[Binarized Neural Architecture Search](http://arxiv.org/abs/1911.10862)|AAAI2020(1911) Beihang|-|Task|Darts+Binary|
|[CP-NAS: Child-Parent Neural Architecture Search for Binary Neural Networks](http://arxiv.org/abs/2005.00057)|CVPR2020(2005) Beihang|-|Task|Darts+Binary+Tch/Stu|
|[BATS: Binary ArchitecTure Search](http://arxiv.org/abs/2003.01711)|ECCV2020(2003) Cambridge|-|Task|Darts+Binary|

### Mixed Papers

> most of em from random arxiv scan on 03-20

|Title 📕|Source 🎓|Code 💻|Component 🔨|Property 💠|
|--|--|--|--|--|
|[A Survey on Neural Architecture Search](https://arxiv.org/pdf/1905.01392.pdf)|Arxiv(1905) *Martin* at IBM|-|Survey|-|
|[Accelerator-Aware Neural Network Design Using AutoML](https://arxiv.org/abs/2003.02838)|MLsys20-W Gupta|-|Hardware|NAS4Accelerator|
|[MTL-NAS: Task-Agnostic Neural Architecture Search towards General-Purpose Multi-Task Learning](https://arxiv.org/abs/2003.14058)|CVPR2020 Gao|-|Flow|NAS + MultiTasking|
|[GreedyNAS: Towards Fast One-Shot NAS with Greedy Supernet](https://arxiv.org/abs/2003.11236)|CVPR2020 You|-|Weights-Manager/Evaluator|Improvement of HyperNet|
|[EcoNAS: Finding Proxies for Economical Neural Architecture Search](https://arxiv.org/abs/2001.01233)|CVPR2020 Dong.|-|Weights-Manager|Evaluate Proxy|
|[Improved one-shot NAS by suppressing posteriror fading](http://xxx.itp.ac.cn/pdf/1910.02543v1)|CVPR2020 Li|-|Weights-Manager|Bayesian+One-shot|
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
|[NAS-Count: Counting-by-Density with Neural Architecture Search](https://arxiv.org/abs/2003.00217)|Arxiv(2020) Hu|-|Task|Counting|








