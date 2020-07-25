<!-- TOC -->

- [MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning, Spring 2018](#mit-18065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018)
    - [1. The Column Space of A Contains All Vectors Ax](#1-the-column-space-of-a-contains-all-vectors-ax)
    - [2. Multiplying and Factoring Matrices](#2-multiplying-and-factoring-matrices)
    - [3. Orthonormal Columns in Q Give Q'Q = I](#3-orthonormal-columns-in-q-give-qq--i)
    - [4. Eigenvalues and Eigenvectors](#4-eigenvalues-and-eigenvectors)
    - [5. Positive Definite and Semidefinite Matrices](#5-positive-definite-and-semidefinite-matrices)
    - [6. Singular Value Decomposition (SVD)](#6-singular-value-decomposition-svd)
    - [7. Eckart-Young: The Closest Rank k Matrix to A](#7-eckart-young-the-closest-rank-k-matrix-to-a)
    - [8. Norms of Vectors and Matrices](#8-norms-of-vectors-and-matrices)
    - [9. Four Ways to Solve Least Squares Problems](#9-four-ways-to-solve-least-squares-problems)
    - [10. Survey of Difficulties with Ax = b](#10-survey-of-difficulties-with-ax--b)
    - [11. Minimizing _x_ Subject to Ax = b](#11-minimizing-_x_-subject-to-ax--b)
    - [14. Low Rank Changes in A and Its Inverse](#14-low-rank-changes-in-a-and-its-inverse)
    - [15. Matrices A(t) Depending on t, Derivative = dA/dt](#15-matrices-at-depending-on-t-derivative--dadt)
    - [16. Derivatives of Inverse and Singular Values](#16-derivatives-of-inverse-and-singular-values)
    - [17. Rapidly Decreasing Singular Values](#17-rapidly-decreasing-singular-values)
    - [18. Counting Parameters in SVD, LU, QR, Saddle Points](#18-counting-parameters-in-svd-lu-qr-saddle-points)
    - [19. Saddle Points Continued, Maxmin Principle](#19-saddle-points-continued-maxmin-principle)

<!-- /TOC -->
# MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning, Spring 2018
>从这个标题可以看到这是 Gilbert Strang 在 2018 年 MIT 的又一课程，查矩阵范数的内容时突然发现的，刚好不想看书(人懒)，就先学这个吧。[课程链接：youtube](https://www.youtube.com/watch?v=Cx5Z-OslNWE&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k)。建议安装Dualsub谷歌浏览器翻译插件进行学习。<br>课程主题主要分为四个部分：<br>
<div align=center><img src="picture/课程大纲.png"  width="40%" height="40%"><br>
<div align=left>
<br>

>可以看到，这个系列课程与计算机专业息息相关，是之前线性代数课程的一个延申。并且该课程有一些在线编程类项目作业，而没有期末考试。可以用 Matlab/Python/Julia 等工具/语言完成。好的，现在我们开始走进这门课。<br>
<div align=center><img src="picture/老师.png"  width="70%" height="70%"><br>
<div align=left>
<br>

## 1. The Column Space of A Contains All Vectors Ax
>本节课没有太多内容，主要提到的是两点内容：矩阵的 CR 分解，矩阵乘法的另一种看待方式。<br>
首先，矩阵 A = C·R 分解中，C 代表列空间的基组成的矩阵，其中的基向量是从 A 矩阵中的列向量中原封不动的抽取的，只要是线性无关的就加进来。R 代表行空间的基组成的矩阵，其中的基不是从 A 中直接获得，而是通过一定计算推导而得(课上的矩阵很简单，就直接看出来了)。特别的，我们发现 C 的最后一行中的 5 7 如果和 R 矩阵的两个行向量进行组合恰好就是 A 的第三行，这其实不是巧合，也许之后我们会再次接触这样的内容：<br>
<div align=center><img src="picture/CR分解.png"  width="50%" height="50%"><br>
<div align=left>
<br>

>矩阵乘法 A = B·C (A 为 m * n 矩阵)我们之前总用 B 的行向量点乘 C 的列向量来得到。然而，还有一种看待方式是用 B 的第 k 列向量乘 C 的第 k 行向量来得到矩阵(k = 1,2,...,n)，之后把各个矩阵相加，这样的计算次数和之前是相同的。<br>
<div align=center><img src="picture/矩阵乘法另一种方式.png"  width="80%" height="80%"><br>
<div align=left>
<br>

## 2. Multiplying and Factoring Matrices
>开始教授提到了五种之前讲过的矩阵分解方式(Factoring)。之后讲了一些关于秩1矩阵分解，矩阵乘法，四个基本子空间的内容，之前也都提到过，在此不再赘述。：<br>
<div align=center><img src="picture/矩阵分解.png"  width="60%" height="60%"><br>
<div align=left>
<br>

## 3. Orthonormal Columns in Q Give Q'Q = I
>本节课的重点在正交矩阵，教授开始讲了旋转矩阵的正交性，之后又引出了反射矩阵(det为负的情景)的正交性，Householder矩阵的正交性：<br>
<div align=center><img src="picture/反射矩阵.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>Hadamard矩阵正交性：<br>
<div align=center><img src="picture/Hadamard.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>Haar小波矩阵(图像压缩中讲过)正交性(可进一步将列向量单位化)：<br>
<div align=center><img src="picture/Haar.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>傅里叶矩阵(四阶)正交性：<br>
<div align=center><img src="picture/傅里叶.png"  width="50%" height="50%"><br>
<div align=left>
<br>

## 4. Eigenvalues and Eigenvectors
>这节课首先需要注意的是矩阵 AB 和 BA 的相似关系。AB 和 BA 相似，相似矩阵 M = B，因此 AB 和 BA有相同的特征值，但是 A 的特征值乘 B 的特征值往往不等于 AB 的特征值，同样 A 的特征值加 B 的特征值往往不等于 A + B 的特征值：<br>
<div align=center><img src="picture/AB和BA.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>其次就是，实对称矩阵具有实特征值，实对称矩阵的正交分解：A = Q·Λ·QT，也称之为谱定理。其他内容就是一些回顾性内容了。

## 5. Positive Definite and Semidefinite Matrices
>课堂主要是一些关于正定矩阵和半正定矩阵的验证，以及一些回顾性内容，下图列出了五个判定正定的条件，半正定往往只需要稍作修改即可(大于变成大于等于)：<br>
<div align=center><img src="picture/正定半正定.png"  width="70%" height="70%"><br>
<div align=left>
<br>

## 6. Singular Value Decomposition (SVD)
>之前我们也花了很大篇幅来说明 SVD，主要是运用 A·AT 和 AT·A 的特性去计算分解后两个正交矩阵和特征值的值。但是实际中往往 A 会很庞大，A·AT 的计算量很大，因此我们不常用这种方法。而是，用下图的方法来求得另一组正交矩阵 U ，我们常常是很容易得到矩阵 A 和一组正交矩阵 V，还有特征值。这一组式子来自于我们最初讨论 SVD 时，把一组单位正交基 V 转化为另一组单位正交基 U 的过程。<br>
<div align=center><img src="picture/奇异值分解简便.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>而通过证明，我们发现这样得到的矩阵 U 确实是正交矩阵：<br>
<div align=center><img src="picture/U的正交.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>当我们分解过后，我们来看 SVD 的几何意义(如下图所示)，当一个矩阵 A 作用到一个向量 x 上时，可以看作 U·Σ·VT 三个部分分别作用的结果。VT 为正交矩阵，作用类似于旋转/反射，Σ 对角矩阵作用后相当于拉伸/压缩成一个椭圆，最后正交矩阵 U 作用也是一个旋转。这个示意图是二维情况，并且我们假设了 Σ 是正定的，不存在升维/降维情况。整个过程实际上涉及了四个参数，首先 VT 的旋转/反射涉及一个参数 θ 。Σ 是二维的，因此涉及两个参数，对应两个基的拉伸幅度。最后 U 与 VT 类似：<br>
<div align=center><img src="picture/奇异值分解几何意义.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>如果扩展到三维，那三个矩阵就分别对应3、3、3个参数，加起来是 9 个，旋转此时也涉及三个方向，三个角。在实际应用中我们常常关注奇异值分解的第一个分量的值 σ1，因为 Σ 矩阵上的奇异值往往按从大到小排列，展开式中每一项的秩为 1 (如第一项 u1·σ1·v1T)：<br>
<div align=center><img src="picture/奇异值分解展开式.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>之后从奇异值分解引出了极分解(Polar Decomposition)，**极分解告诉我们任意一个矩阵 A 都可以分解为一个对称矩阵和一个正交矩阵的乘积**。 A = S·Q，我们可以由奇异值分解进行推导：A = U·Σ·VT = (U·Σ·UT)·(U·VT)，前半部分很明显是对称的，后半部分是两个正交矩阵的乘积，还是正交矩阵(由正交矩阵几何意义很容易想到)。

## 7. Eckart-Young: The Closest Rank k Matrix to A
>本节课的重要思想就是关于主成分分析(PCA)的核心观念，我们如何找到最接近 A 的秩为 k 的矩阵？首先我们需要回答：如何度量两个矩阵的接近程度？答案是：范数。关于向量范数我们之前已经有所耳闻，主要有以下三种最关键的，下面的两个公式是范数需要满足的条件：<br>
<div align=center><img src="picture/向量范数.png"  width="50%" height="50%"><br>
<div align=left>
<br>

>之后我们引出矩阵范数，其中 σ 就是矩阵奇异值分解中的奇异值，这是最主要的三种矩阵范数。其中 Frobenius 范数还有下面图的表示方法(关于奇异值的部分)：<br>
<div align=center><img src="picture/矩阵范数.png"  width="50%" height="50%"><br>
<div align=left>
<br><div align=center><img src="picture/Frobenius.png"  width="80%" height="80%"><br>
<div align=left>
<br>


>接下来，就是 Eckart-Young 的发现(如下图所示)，他发现既然任何矩阵 A (秩为r) 都可以奇异值分解为如下 r 个秩 1 矩阵相加的结果，那我们取前 k 个构成的矩阵，就是最接近 A 的秩 k 矩阵。其中 σi 是从大到小排列的，这个“最接近”的度量方式就是上面我们提到的三种矩阵范数其中的任意一个 ：<br>
<div align=center><img src="picture/秩k最接近A.png"  width="50%" height="50%"><br>
<div align=left>
<br>

>最后是关于 PCA 的内容，教授举了一个很像最小二乘的例子(图之后重新画。。。)：<br>
<div align=center><img src="picture/LSE和PCA.png"  width="70%" height="50%"><br>
<div align=left>
<br>

>左图是最小二乘估计(Least Square Estimate)的方法，右图是 PCA 的方法，我们发现二者度量的误差距离是不一样的，LSE 是竖直的距离，而 PCA 度量的是垂直于直线的距离。并且这是一个直线拟合的问题，PCA 的结论告诉我们那个斜率 k 就是所有样本构成的 2 * n 矩阵经过奇异值分解后(n为样本数，矩阵第一行代表年龄，第二行代表身高)，奇异值 σ1 的值。

## 8. Norms of Vectors and Matrices
>本节课我们首先来看二维向量范数，下图从左至右依次是各个范数等于 1 的情况：2-范数为1、1-范数为1、无穷范数为1、0-范数为1 (0-范数指的是向量中不为 0 的分量个数)。有趣的是，我我们从 0-范数 一直增加到无穷范数我们发现图像是一个扩张的过程，从两条线扩张到菱形再到圆再到方形。而一个好的范数，我们要求图像是凸的而不是凹的，如 1、2、∞ 范数都是凸的，而 p ∈ [0,1) 的范数都是凹的：<br>
<div align=center><img src="picture/二维向量范数.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>补充一点就是 S-范数 (课本中也叫加权范数/椭圆范数)。S-范数定义为 ||x|| = (xT·A·x)^1/2，我们也可以按照上图作出此图像 S-范数等于 1 的情况。下图是一个正定矩阵的 S 范数为 1 的情况，做出图来就是一个椭圆：<br>
<div align=center><img src="picture/S范数.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>当我们进行空间中点的范数大小判定时，我们就可以用对应的图形从原点进行扩张，先命中的就是范数更小的点。下图可能不太清晰，但是也标出了图中直线上 L1，L2 范数最小的两个点。其中判定 L1 最小点时，我们用菱形扩张，L2 最小点我们用圆进行扩张：<br>
<div align=center><img src="picture/L1L2范数.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>下面我们详细介绍一下矩阵范数，下图右侧是矩阵 2-范数 的一个定义。我们看到是从向量范数角度进行定义的， ||Ax||/||x||，这个值可以看作向量 x 经过 A 矩阵作用后改变的程度，我们叫这个数为爆炸因子，而 2 范数就是找一个最大的爆炸因子，这就和向量有关。如果 x 是 A 的特征向量，那么此时对应的爆炸因子就是相应的特征值，但是对于一个一般的矩阵(非方阵)，这个值往往是奇异值 σ ，而最大的爆炸因子就是 σ1，推导过程见右下，奇异向量 ||v|| = 1，而奇异值分解中 A·v = σ·u，||u|| = 1：<br>
<div align=center><img src="picture/矩阵2范数来源.png"  width="80%" height="80%"><br>
<div align=left>
<br>

## 9. Four Ways to Solve Least Squares Problems
>前半节课教授回顾了关于四个基本子空间和伪逆的内容。<br>
>关于之前的最小二乘问题，我们在前面也得出了很多结论。如 A·x = b 无解时，我们可以两边同左乘 AT 则原方程有解，得到 AT·A·x-hat = AT·b，当是我们是用投影的角度来看待这个方程有解的。现在我们有另一种看待方式，就是在无解方程 A·x = b 上，最小化 ||Ax-b||^2，这里我们取 2-范数 的平方，之后对 xT 求导，就可以得到 AT·A·x-hat = AT·b 这个式子：<br>
<div align=center><img src="picture/最小二乘另一角度.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>上面两个角度我们都假设了 A 是列满秩的，如果列不满秩，则上述方法不可行。同时，当列满秩时，我们发现，上述方法解出 x-hat = (AT·A)^-1·AT·b = A+·b ( A+ 为伪逆)。即当列满秩时，(AT·A)^-1·AT = A+ 右乘 A ，左右两式都得到单位矩阵 I (n * n)。这就是第三种方法，但 x-hat = A+·b 这种方法在列不满秩情况下也可使用：<br>
<div align=center><img src="picture/最小二乘3.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>至于第四种方法下节课我们会提到，如下图所示。第一种就是伪逆方法，第二种应该是上面介绍的最小化 Loss = ||Ax-b||^2 然后求导的方法，第三种施密特正交化应该是投影的方法(投影的根源就是正交)，第四种下节课我们会提到：<br>
<div align=center><img src="picture/最小二乘4方法.png"  width="60%" height="60%"><br>
<div align=left>
<br>

## 10. Survey of Difficulties with Ax = b
>本节课我们专注于现实的应用问题，在面对现实问题时，我们要求解 A·x = b 此种问题时，往往会遇到 A 不是方阵但列满秩的情况，上一节课也提到过我们可以用那四种方法来解决(最小二乘问题本质上就是 Ax = b 的求解问题)。但是还有一种情况是 A 是一个非奇异矩阵，但 nearly singular 很接近奇异，也就是说有的奇异值很接近 0 ，或者说 A 的逆矩阵中有值很大的元素，这样我们求解出来的结果往往不太符合期望。此时我们可以加一个正则项，如下图所示：<br>
<div align=center><img src="picture/最小二乘4.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>我们需要最小化的 Loss 改变了，而这个 A*·x = b*，对应的最小二乘方法就是上节课我们提到的方法 4 中的那个等式 (AT·A+δ^2·I)x = AT·b (至于这个推导原因我也不清楚，A 为 1 * 1 矩阵时是一个求导的关系，即最小化的式子求导为 0 得到的式子恰好是 A*·x = b* 的结果)。教授给我们推导了 A = [σ] 即 A 为 1 * 1 矩阵的优化解决方式：<br>
<div align=center><img src="picture/1乘1最小二乘.png"  width="60%" height="70%"><br>
<div align=left>
<br>

>我们可以通过最小化那个式子(求导)，或者解  (AT·A+δ^2·I)x = AT·b 这个等式，来得到同一个式子。然后结果 x 在上图的左下角。我们发现如果我们的正则项系数 δ 取 0 ，那么当矩阵 A 为 0 时，即 σ = 0 时，我们的 x 会出现错误(分母为 0)，但是如果我们有正则项，就算 σ = 0 ，我们也能得到两个结果(见上图右下角)。因此正则项不但保证了逆的不理想的情况，也保证了所有情况有解。而这个解的形式最终会和 A+ 相近，即：<br>
<div align=center><img src="picture/正则项结果.png"  width="70%" height="70%"><br>
<div align=left>
<br>

## 11. Minimizing _x_ Subject to Ax = b
>这节课讲的一点是解决施密特正交化的一些问题。之前我们讨论过施密特正交化是依次取基向量的误差部分 e ，但是当两个向量十分接近时，e 就会很小，我们不希望这种事情发生。于是，我们在选取基向量时，可以依次考察所有向量，选择一个最适合做下一个正交基向量的向量，即误差最大的来进行迭代。这样看似计算量变大了，其实时几乎不变的，因为我们在用之前的方法时，也是需要依次计算的。如下图，我们在挑选 q2 时，选取剩下所有向量与 q1 进行误差 e 的计算：<br>
<div align=center><img src="picture/施密特正交化改良.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>还有最后提到 Krylov 子空间迭代。

## 14. Low Rank Changes in A and Its Inverse
>本节课主要提到了在实际问题中会遇到的矩阵秩 k 扰动问题。我们常常会先解决一个类似 Ax = b 的问题，但是随着时间变化，我们收集到更多的数据，或者改动以往的数据，那么此时的 A 发生变化，当然问题的解 x 也会发生变化。我们如果不知道矩阵扰动的相关公式，那么就得从头开始重新计算一遍。但有了相关公式，我们就可以根据之前的解和扰动因素得到新的解，大大降低了运算代价。<br>
>教授主要讲了三种矩阵扰动的公式，如下图所示，第一种是对单位矩阵 I 秩为 1 的扰动，第二种是对单位矩阵 I 秩为 k 的扰动，第三种是对任意矩阵 A 秩为 k 的扰动，只要我知道了他们的逆矩阵如图所示则新结果 x 也就得出了：<br>
<div align=center><img src="picture/矩阵扰动.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>我们依次来看相应的公式。第一种如下图，我们可以用逆矩阵乘本身的技巧来验证这个是正确的：<br>
<div align=center><img src="picture/扰动1.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>第二种如下图，我们可以用相同的验证方式：<br>
<div align=center><img src="picture/扰动2.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>第三种如下图，这种就类似我们实际中的问题了：<br>
<div align=center><img src="picture/扰动3.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>教授给我们举了最小二乘的动态问题，即不断有新样本加入的问题，如下图所示，我们在原本 AT·A·x = AT·b 的基础上给 A 增加了更多的行 vT (样本)，而此时我们问题的构造就如下图一样，也就是给 AT·A 了一个秩为 r(v·vT) 的扰动，而要求得 x-new 的值。这个问题类似于卡尔曼滤波器，如果之后碰见再细究：<br>
<div align=center><img src="picture/最小二乘扰动.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>而扰动问题的解决如下图所示，我们常常在解决一个带有扰动的新问题时(第二行)，已经解决了一个基础问题 A·w = b ，而此时如果我们能解决 A·z = u，那么就可以用已知的解来组合为新问题的解，新问题的解的形式见第二张图(这个形式好像有点问题)：<br>
<div align=center><img src="picture/扰动问题.png"  width="70%" height="70%"><br>
<div align=left>
<br>
<div align=center><img src="picture/扰动解.png"  width="40%" height="40%"><br>
<div align=left>
<br>

>BTW，这一节的公式推导都没有给出，只是对于前两个关于单位矩阵 I 的扰动公式的正确性进行了验证，这个也很好验证。

## 15. Matrices A(t) Depending on t, Derivative = dA/dt
>这节课开头，教授给我们讲了 A^-1 的导数与 A 的导数的关系，如下图所示。先构建一个恒等式 B^-1 - A^-1 = B^-1·(A - B)·A^-1，然后我们把 B 看作 A 矩阵的增量矩阵，即 B = A + ΔA，再两边除以 Δt 可以得到下面的式子。注意，如果 B = A + ΔA 那么 B^-1 - A^-1 在意义上就能表示 ΔA^-1 ，就像 B - A = ΔA 一个道理：<br> 
<div align=center><img src="picture/A逆导数.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>之后我们再让 Δt -> 0，就得到了微分的形式。我们可以把 A 当作 1 * 1 矩阵 [t] 来验证正确性，即其中 A^-1 = 1/t：<br> 
<div align=center><img src="picture/微分形式.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>现在我们解决 dλ/dt 的问题，如下图。其中矩阵 A、A的特征值(λ)、A的特征向量(x)、A行空间的特征值(与 A 特征值相同)、A 行空间的特征向量(y)，都和 t 有关。我们可以得到第一行的 Fact ，其中后面那个的写法不常见，但是确实有如此写法来描述行空间特征向量和特征值关系的公式(我不知道为什么)。然后，yT·x = 1 的原因下面有一个评论说是因为 Matlab/人工 在处理时，往往对特征向量进行归一化，之后我们对 Fact2 两边同右乘 x(t) 得到 Formula 1(最后一行)，经过化简恰好是 λ(t) ，然后就可以对其求导：<br> 
<div align=center><img src="picture/λt.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>求导结果如下，神奇的是第一项和第三项相消为 0 (没看懂)，结果只剩下第二项就是 dλ/dt 的值：<br> 
<div align=center><img src="picture/λt求导结果.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>最后又讲了一些关于 interlacing theorem 交错定理的内容，但是我听的半懂不懂，并且似乎不是很重要，我就不放上笔记了。

## 16. Derivatives of Inverse and Singular Values
>承接上节课，教授讲了 dσ/dt 的公式，和 dλ/dt 的证明方法类似，得出的结论也类似：<br> 
<div align=center><img src="picture/dσdt.png"  width="40%" height="40%"><br>
<div align=left>
<br>

>证明过程如下，和 dλ/dt 的开头类似，教授用一个三项的乘积代表 σ(t) ，这是从 SVD 中的 Av=σu 中得来的，之后用求导法则求 σ(t) 的导数，最后只留下中间项，第一项第三项都为 0 ，并且我们看 uT·u = 1 的导数（最右边），(duT/dt)·u 和 uT·(du/dt)这两项其实是相等的，所以 du/dt = 0。同理：dv / dt = 0，所以第一项第三项为 0 ：<br> 
<div align=center><img src="picture/dσdt证明过程.png"  width="80%" height="80%"><br>
<div align=left>
<br>

## 17. Rapidly Decreasing Singular Values
>本节课是由 Alex Townsend 讲述，主要讲述关于低阶矩阵的问题。首先为我们陈述了三个关于矩阵奇异值的 Fact 。如果矩阵 x 是 n * n 方阵，前 k 个奇异值大于 0，后面的都小于零，则有：1. 矩阵 x 的秩为 k；2. 矩阵 x 可以分解为 k 个秩 1 矩阵相加的形式；3. 矩阵的行秩和列秩都为 k：<br> 
<div align=center><img src="picture/奇异值Fact.png"  width="70%" height="70%"><br>
<div align=left>
<br>

>于是，Alex 给我们举了各个国家的国旗例子。我们可以看到右上角第一行的国旗如果用低秩矩阵展开的形式(上面第二个 Fact)发送的话应该都很容易，因为他们有很多相同的行，而第二行的如日本国旗似乎就不那么理想，第三行含有三角的是最不理想的。因此当我们碰到低秩矩阵时，用展开形式发送简单，而高秩形式就直接发送原矩阵更快捷。至于低秩的具体数字描述见下图第一行，因为我们如果把整个n阶矩阵发送的话，开销为 n^2 ，而秩为 k 的矩阵展开我们有 k 项，每一项有 2n 个条目，因此开销为 2kn，如果 2kn < n^2，我们就称为低秩。但在实际操作中，我们往往要求 k << n/2：<br> 
<div align=center><img src="picture/三角旗.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>上图中的三角旗为什么发送开销大呢？Alex 给我们推出了类似左下角形式的下三角 n 阶矩阵的相关逆矩阵，我们由 (XT·X)^-1，就能得出矩阵 X 奇异值相关信息，在最后一行 Alex 给我们写出了三角旗矩阵 X 的奇异值，我们发现都不为 0 ，并且不收敛到 0。归一化后画出了奇异值图像，横坐标是第 i 个奇异值，纵坐标是归一化后各奇异值大小。

>之后，就到了圆形旗如日本旗那样，如下图所示。我们可以从旗的中间圆形掏出一个正方形，这样正方形的秩就为 1，剩下的部分就是轴对称图形，我们可以取 1/4，秩是和原来相同的，这 1/4 可以分为两个部分，我们分别计算二者的秩，最后结果取决于图中圆的半径 r，大约下来是 1/2·r-1 也不会很大：<br> 
<div align=center><img src="picture/日本旗.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>接下来介绍了数字秩(Numerical rank)的定义，其中 ε 是一个容忍度，我们可以看到 Numerical rank 关联的是奇异值，而不是特征值。而当 ε = 0 时，Numerical rank 就等于 rank 了：<br> 
<div align=center><img src="picture/数字秩.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>在实际中有很多 Numerical low rank 矩阵，如下图所示的 Hilbert 矩阵和 Vandermonde 矩阵，这两个矩阵其实是满秩的，但是 Numerical rank 很低。从定义中我们能推测出 Numerical rank 往往会筛出奇异值很低的部分。Numerical low rank 矩阵不总是好的，因为他们的逆矩阵很难求：<br> 
<div align=center><img src="picture/低数值秩矩阵.png"  width="60%" height="60%"><br>
<div align=left>
<br>

## 18. Counting Parameters in SVD, LU, QR, Saddle Points
>本节课我们逐渐向概率、优化、深度学习进行过渡。首先，教授给我们讲了在计算中各种矩阵分解的参数个数，如下图所示，下面我依次进行推导：<br> 
<div align=center><img src="picture/矩阵参数个数.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>首先我们从单个矩阵开始看。<br>
L 代表单位上三角矩阵，对角线元素都为 1，因此对角线 元素不算参数，参数个数为：1+2+...+(n-1) = 1/2·n·(n-1)。<br>
U 代表下三角矩阵，但是对角线元素不设限，因此参数个数为：1+2+...+n = 1/2·n·(n+1)。<br>
Q 代表正交矩阵，正交矩阵的特点是列向量标准化，且两两正交；因此我们选择 Q 的第一列时只需要 n-1 个参数，第 n 个值只需要利用标准化性质计算可得。第二列除了标准化性质，还有与第一列正交性质，因此只需要 n-2 个参数。第三列除了标准化性质，还有与第一、二列正交性质，因此只需要 n-3 个参数，以此类推。总参数个数为：(n-1)+(n-2)+...+(1) = 1/2·n·(n-1)<br>
Λ 代表对角矩阵，当然有 n 个参数。<br>
X 代表特征矩阵，计算机在处理时往往有个约定，就是特征向量第一个值为 1 ，这个可以通过对特征向量乘以一个合适的常数 c 来得到。因此参数总数为 n^2 - n。<br>

>下面我们开始看分解矩阵的参数个数。<br>
A = LU ，L、U两个矩阵可以由上面分别得到参数个数，总的参数个数为：1/2·n·(n-1)+1/2·n·(n+1)=n^2<br>
A = QR ，Q 是正交矩阵，R 是上三角矩阵但是对角线不限制，因此，总的参数个数为：1/2·n·(n-1)+1/2·n·(n+1)=n^2<br>
A = XΛX^-1，X 和 X^-1只需要取一个另一个就知道了，总的参数个数为：n^2 - n + n = n^2<br>
S = QΛQT，Q和QT只需要取一个另一个就知道了，总的参数个数为： 1/2·n·(n-1) + n = 1/2·n·(n+1)<br>
A = QS (polar decomposition极分解)，Q是正交矩阵，S是对称矩阵，总的参数个数为：1/2·n·(n-1)+1/2·n·(n+1) = n^2<br>
A = U·Σ·VT，这个比较麻烦，我们用下面的图进行解释。我们首先假设矩阵 A 是 m * n 的，并且行满秩，即 r(A) = m，并且 m < n。然后对于 U、Σ、VT 的尺寸见下图所示，U 是秩为 m 的正交矩阵，当然参数个数为：1/2·m·(m-1)。Σ 秩为 m ，当然参数个数为：m。最后 VT 有些特别，虽然它是 n * n 的但是它只有 m 列是未知的，且相互标准正交，这部分的参数个数为：(n-1)+(n-2)+...+(n-m)。剩下的 n - m 列是来源于零空间正交的标准基向量，因此我们不关心它们的值。所以这三部分加起来恰好是 m * n 个参数。<br> 
<div align=center><img src="picture/奇异值分解参数.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>对于 SVD，更加 general 的情况是 rank(A) = r < m < n。此时，分解矩阵的各部分尺寸见下图所示，此时第一部分参数个数为：(m-1)+(m-2)+...+(m-r)，对角矩阵参数个数为：r，第三部分参数个数为：(n-1)+(n-2)+...+(n-r)，加起来参数个数是: mr+nr-r^2<br> 
<div align=center><img src="picture/SVD秩r.png"  width="40%" height="40%"><br>
<div align=left>
<br>

>接下来介绍关于拉格朗日乘数法的内容，如下图。我们在遇到求 min 1/2·xT·S·x 的问题时，往往会遇到 Ax = b 的限定。于是我们构造 L(x，λ) 分别对 x，λ 求导，得到右下角的矩阵形式方程组。这样求出来的 x 就是鞍点，而不是最值点，因为在 x 方向是最值，但是有 λ 项的约束。鞍点（Saddle point）在微分方程中，沿着某一方向是稳定的，另一条方向是不稳定的奇点，叫做鞍点：<br> 
<div align=center><img src="picture/拉格朗日乘数法.png"  width="80%" height="80%"><br>
<div align=left>
<br>

>鞍点的图像如下：<br>
<div align=center><img src="picture/鞍点.png"  width="40%" height="40%"><br>
<div align=left>
<br>

>而对于上述矩阵形式方程组我们能做什么呢？我们可以进行如下变换，于是我们就可以得知主元正负的情况(我也不知道这一步的目的是在干嘛)：<br> 
<div align=center><img src="picture/拉格朗日乘数法相关变换.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>我们还可以引出瑞利熵(Rayleigh Quotient)：我们看到 R(x) 的最大值就是矩阵 S 最大的特征值 λ1，x 为相应的特征向量 q1，最小值就是矩阵 S 最小的特征值 λn，x 为相应的特征向量 qn。在这之间的其他一些特征值和对应的特征向量得到的 R(x) 就是鞍点：<br> 
<div align=center><img src="picture/Rayleigh.png"  width="60%" height="60%"><br>
<div align=left>
<br>

## 19. Saddle Points Continued, Maxmin Principle
>承接上一节，我们讲一下为什么其他特征值对应的特征向量得到的 R(x) 就是鞍点。我们先看一下下面的一个例子。我们取 S 为一个正定矩阵，R(x) 展开的形式也呈现在板书上，特征值以及各特征向量见左下角。其中每个特征向量点对应的 R(x) 的一阶导数都为 0 ，可以当作一个 Fact：<br> 
<div align=center><img src="picture/鞍点2.png"  width="60%" height="60%"><br>
<div align=left>
<br>

>现在我们研究各个特征值，首先最小值和最大值是显而易见的，而位于中间的特征值 λ2 = 3，有什么特点呢？下面的板书有点混乱，我来解释一下。首先，我们的结论是：λ2 是 R(x) 的一个鞍点，并且在某一个二维子空间内是最大值点，在另一个二维子空间内是最小值点。我们取子空间 (u,v,0) ，如右下角所示，这是 λ1 对应特征值 (1,0,0) 和 λ2 对应特征值 (0,1,0) 张成的子空间，在这个子空间内 R(x) 的最小值为 3 ，对应的 R(x) 展开式见右上角；但在 λ2 对应特征值 (0,1,0) 和 λ3 对应特征值 (0,0,1) 张成的子空间 (0,v,w) 内，R(x) = 3 是最大值：<br> 
<div align=center><img src="picture/中间鞍点.png"  width="60%" height="60%"><br>
<div align=left>
<br>