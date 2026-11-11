# 当前SFRUNet架构示意图

架构图：architecture.png。使用内置image_gen工具生成，并进行一次输入连线修订。
工具接口由运行环境提供图像模型，具体模型版本由该服务配置。

核验依据：Experiment/SFRUNet.py、Experiment/models.py、Experiment/cell_sfr_objective.py，
以及Results/Exp2_LOO_Fiducial_SFRUNet_ReLU_Candidate1_seed43_20261107_041950_b08b68f2/config.json。

图示为当前ReLU Candidate1配置：12物理通道与18 Fourier通道拼接，特征通道32/64/128，
E1/E2/Bottleneck/D2/D1各两个残差块，两个对应尺度的Concat跳跃连接。
物理输出依次为Delta=s_C*sinh(z)、q=B+Delta、Y_hat=ReLU(q)、C_hat=Y_hat-B、总SFR=sum(Y_hat)。
损失为最终逐格MSE、最终总量MSE、候选逐格MSE，三项权重均为1；验证评分为前两项之和。
全部损失采用训练集估计的尺度。ReLU与物理输出组合参与可微训练。
残差块使用SE通道注意力；固定R_h使用conditioning=none。

已目视核验输入Concat、U形多尺度路径、各特征shape、两条Concat跳连、ReLU位置及损失公式。
本次交付为架构说明图片，未修改模型源码或启动训练。
generation_prompt.txt保存主生成提示词原文。
