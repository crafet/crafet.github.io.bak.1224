---
layout: post
title: 理解在线广告ctr预估的AUC
---

标签： 在线广告 ctr预估 机器学习 模型 AUC

---

## 广告ctr预估背景

ctr（click through rate),即点击率，是在线广告，特别是效果广告关注的一个重要指标。高点击率对于广告主以及广告平台本身都是乐意看到的：对于广告主而言，意味着可能会有进一步的转化（受众点击了广告，注意力转移到广告的landing page，可能会购买广告主的商品等）；对于广告平台来说，一次有效的click也意味流量得到了变现。

然而，广告的点击率一直是比较低，对于传统的效果广告而言，不可见曝光的ctr大概在1/1000左右，对于可见曝光的广告，其点击率则是在1/100左右，可见多数的曝光是没有click发生的。因此，投放一个可能会被高概率点击的广告是非常重要的，一方面有效利用了本次曝光，同时一方面也实现了流量变现。

因此，
> 对即将展现的广告完成准确的点击率预估显得尤为**重要**。

## ctr 预估过程

在线广告的ctr预估，在工业界有成熟的解决方案。即利用历史log，进行训练model，线上server加载这个model。当有广告展现请求时候，server根据请求的上下文信息计算出候选广告的预估点击率，从其中选择出最高预估ctr进行展现。

> 历史数据包括展现日志以及点击日志，join之后，做数据精华后形成正负样本集，基于Logistic Regression（LR)训练算法，计算出feature的权重。
常见的feature有:
> > 广告主特征:如行业分类
> > 受众特征：如cookie，性别等
>> 广告自身特征:如物料的类型（文字链、图片、flash等类型）、广告的核心词等。


LR 训练算法基于历史数据，迭代计算出这些特征权重，并加载到线上参与计算。

## AUC在特征选择上的作用
在优化模型的时候，我们期望能够加入足够多有典型区分度的特征。特征有良好的区分度，有助于在筛选广告阶段进行准确的排序。准确的排序意味着，在排序好的候选广告中，可以选择top1，或者top2等这样高预估ctr的广告进行展现。一项*eye tracking study* [^eye tracking study]中给出一份结论：对于同一个网页上的广告位而言，用户在浏览页面的时候，从页面上方到下方，广告位的点击率是**骤降**的，甚至第二位的position，其ctr相对第一position的ctr会下降90%。

在这种情况下，假定候选广告按照预估ctr(pCTR)进行降序排列后，top前2个广告（命名为a、b）需要选择出进行展现。一个特征的加入或者缺失，都会影响到a、b的排名。假设a的收益更大，如果排序为b、a，那显然a的ctr会下降较多，而b的收益又较小，那总体上，**本次广告的曝光展现并不是最优的**。所以，任何一个特征的引入或者删除，都要基于实际的历史样本来进行评估这么做的影响，而AUC恰好是用来量化这种影响的重要指标，或者说AUC是用来评估模型排序能力的重要指标。

在介绍AUC之前，需要引入两个前提概念：ROC以及confusion matrix。
下面先介绍ROC。

> ROC的全名叫做Receiver Operating Characteristic，其主要分析工具是一个画在二维平面上的曲线——ROC curve。平面的横坐标是false positive rate(FPR)，纵坐标是true positive rate(TPR)

ROC最早是用在传统医疗工业上的度量指标，后来在模式识别（pattern recognition)、machine learning中被引入作为衡量分类器优劣的判断指标。一个典型的ROC曲线如下：

![ROC曲线例子](http://img4.doubanio.com/view/note/large/public/p8947349.jpg)

在实际的使用中，定义ROC下的面积为AUC(area under curve），即曲线下的面积。总面积为单位矩形(1*1)，而AUC的值范围是0~1。一般的，当加入一个高区分度的特征时候，模型的AUC通常是增加的，图片中的虚线下的面积是0.5。即不做任何的排序，完全random出值后进行排序，因此通常一个合理的model的AUC应该是大于0.5的。当然model的AUC也可以是小于0.5，这种情况下，可以理解为*model有预见性的避开了正确答案:)*

AUC的计算，需要引入confusion matrix（混淆矩阵）。

> 混淆矩阵在计算分类recall、accuracy的时候常常用到。
一个典型的两类混淆矩阵如下：
 |  正样本个数P   |  负样本个数N  | predict |
| :-----:           | :----:        | :------:   |
| true positive |   false positive     |Ｙ|
| false negative |  true negative |Ｎ|

> * true positive : 预估为正样本positive，并且预测正确true
> * false positive: 预估为正样本(positive)：但是预测错了(false)，即**误报**
> * false negative：预估为负样本(negative)，但是预测错了(false)，即**漏报** -- *所谓漏报是站在正样本的角度来看，漏报了正样本*。
> * true negative：预测为负样本（negative），并且预测正确（true）。

在AUC的计算中，我们站在正样本的角度来看问题，即使用相关的两个数据项，true positive以及false positive。在上述的分类结果下，正样本都被分正确的比率是 $$tpr = \frac {tp} {P}$$，而false positive的比率为$$ fpr = \frac {fp} {N} $$

将所有广告按照pCTR降序排列，再结合实际的样本数据，就可以计算AUC了。

> [[这篇文章]](http://www.douban.com/note/284051363/) 中提到一个如何使用fpr与tpr来计算AUC。

> *In signal detection theory, a receiver operating characteristic (ROC), or simply ROC curve, is a graphical plot which illustrates the performance of a binary classifier system as its discrimination threshold is varied*

其中提及的**discrimination threshold**，~~直译为"区分阈值"。~~ 指的是，每次在一个`pCTR`上进行划分，取划分这个`pCTR`为阈值，高于这个阈值的是预估的正样本，低于这个阈值，是预估的负样本。在这种情况下，计算当前这个划分的`tpr`和`fpr`。得到一个点对（`fpr`， `tpr`）。每个划分都会得到一个点对，那么就可以绘制出ROC,进而计算出AUC。
| 是否click（1为click，为正样本；0为负样本） | pCTR|
| :-----: | :------:|
|`1`|0.9|
|`1`|0.8|
|0|0.7|
|`1`|0.5|
|0|0.4|
|`1`|0.3|
|0|0.2
|0|0.1|

为了计算简答，给出了8个样本，4个正样本，4个负样本。(`当然，实际情况下，正负样本的比例远比这个大的，是imbalance的`)为了区分，其中`1`表示实际的样本是发生了click，而0则表示实际情况下并没法有产生click。而pCTR为预估的ctr。我们的期望是**我们的预估的ctr高的广告，其实际应该有click，这样就说明我的预估是有效且准确的，当然没有click发生也是可以理解，预估目前还做不到100%的准确。**

其划分区间的方式是：
> * 在0.9划分，认为>=0.9的都是正样本，<0.9的都是负样本。次划分下，$tpr = \frac 1 4$，$fpr = \frac 0 4$。
> * 继续，在0.8处划分，认为>=0.8的都是正样本，即会高概率发生click的，<0.8则预估为负样本，此划分下, $tpr = 
\frac 2 4$,$fpr = \frac 0 4$。
> * 继续，在0.7处划分，认为>=0.7的预估为正样本，<0.7的为负样本。此时，有1个是误报的。$tpr =  \frac 2 4$, $fpr = \frac 1 4$.
> * ...
> * 以此类推，可以计算出所有的fpr、tpr，按照这些点对画出ROC曲线即可。当样本数量足够多情况下，ROC曲线就越平滑。计算的时候将曲线下的每个"小矩形"进行面积累加即可。

回过头来，再"感性"感受下AUC的度量排序能力。理想的排序能力是**按照pCTR进行预估降序排列后，`所有的正样本都排在负样本的前面 --即预估会排在前面的广告都会被click`，此时靠前面的几次划分fpr都是$fpr = \frac 0 N$**。

模型优化的目标是就是努力向这个理想的目标靠拢。


而在实际的AUC计算中，往往会采用另外一种更简单的AUC计算方式，[这篇文章](http://www.cnblogs.com/guolei/archive/2013/05/23/3095747.html) 中提到
> *一个关于AUC的很有趣的性质是，它和Wilcoxon-Mann-Witney Test是等价的。这个等价关系的证明留在下篇帖子中给出。而Wilcoxon-Mann-Witney Test就是测试任意给一个正类样本和一个负类样本，正类样本的score有多大的概率大于负类样本的score。有了这个定义，我们就得到了另外一中计算AUC的办法：得到这个概率。我们知道，在有限样本中我们常用的得到概率的办法就是通过频率来估计之。这种估计随着样本规模的扩大而逐渐逼近真实值。这 和上面的方法中，样本数越多，计算的AUC越准确类似，也和计算积分的时候，小区间划分的越细，计算的越准确是同样的道理。具体来说就是统计一下所有的 M×N(M为正类样本的数目，N为负类样本的数目)个正负样本对中，有多少个组中的正样本的score大于负样本的score。当二元组中正负样本的 score相等的时候，按照0.5计算。然后除以MN。实现这个方法的复杂度为O(n^2)。n为样本数（即n=M+N）*

简单的说，统计下所有正样本排在负样本之前pair/m*n，即$$model_{auc} = \frac {\sum{positiveRank} - \frac {m*(m+1)} 2} {M*N}$$。

这个公式的“感性”解释与前面我们提及的目标是一致的 **--即预估的正样本尽可能是在负样本之前的**，这中情况下依照次公式计算出的值也越大，认为model的排序能力就更好。

##结语
AUC是衡量model的一个重要指标，事实也一些其他指标，如MSE(Mean Square Error)等。这在以后再做总结。


----
##参考
[^eye tracking study]:  https://www.briggsby.com/how-does-google-authorship-impact-ctr/
[2]: http://www.douban.com/note/284051363/