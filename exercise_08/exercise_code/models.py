import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams
        
        ########################################################################
        # TODO: Initialize your encoder!                                       #                                       
        #                                                                      #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 
        # Look online for the APIs.                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Wrap them up in nn.Sequential().                                     #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        #                                                                      #
        # Hint 2:                                                              #
        # The latent_dim should be the output size of your encoder.            # 
        # We will have a closer look at this parameter later in the exercise.  #
        ########################################################################
        
        # 创建编码器网络
        self.encoder = nn.Sequential(
            # 第一层：输入层 -> 512 神经元
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第二层：512 -> 256 神经元
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第三层：256 -> 128 神经元
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 最后一层：压缩到潜在空间维度
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        # self.decoder = None

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################
        # 解码器应该是编码器的镜像结构
        self.decoder = nn.Sequential(
            # 从潜在空间开始
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 128 -> 256
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 256 -> 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 最后一层恢复到原始图像尺寸
            nn.Linear(512, output_size),
            # 使用Sigmoid确保输出在[0,1]范围内，因为MNIST图像像素值在这个范围
            nn.Sigmoid()
        )


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # 确保优化器在初始化时被设置
        self.optimizer = None  # 明确初始化为None
        self.set_optimizer()   # 然后设置优化器

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def set_optimizer(self):
        """设置优化器"""
        # 收集要优化的参数
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        # 确保有参数要优化
        if not any(p.requires_grad for p in params):
            raise ValueError("No parameters to optimize!")
            
        # 确保有正确的学习率
        lr = self.hparams.get('learning_rate', 3e-4)
        if not isinstance(lr, float):
            lr = 3e-4  # 设置默认值
            
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=self.hparams.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # 验证优化器是否正确创建
        if self.optimizer is None:
            raise ValueError("Failed to create optimizer!")


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similarly to the way it is shown in      #
        # train_classifier() in the notebook, following the deep learning      #
        # pipeline.                                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #                                     
        ########################################################################
        # 设置为训练模式
        self.train()
        
        # 获取输入数据
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(self.device)
        
        # 将图像展平
        flattened_images = images.view(images.shape[0], -1)
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        reconstructions = self.forward(flattened_images)
        
        # 计算损失
        loss = loss_func(reconstructions, flattened_images)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        self.optimizer.step()



        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################
        # 设置为评估模式
        self.eval()
        
        # 获取输入数据
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(self.device)
        
        # 将图像展平
        flattened_images = images.view(images.shape[0], -1)
        
        # 无梯度计算
        with torch.no_grad():
            # 前向传播
            reconstructions = self.forward(flattened_images)
            
            # 计算损失
            loss = loss_func(reconstructions, flattened_images)



        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        
        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #   
        # block of fully connected layers.                                     #                                                             
        ########################################################################
        # 获取编码器的输出维度（latent_dim）
        latent_dim = encoder.latent_dim
        
        # 创建分类器模型
        self.model = nn.Sequential(
            # 第一层：从latent_dim到256
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第二层：256到128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 输出层：128到10类（MNIST的类别数）
            nn.Linear(128, 10)
        )
        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################
        # 设置优化器，同时优化编码器和分类器的参数
        self.optimizer = torch.optim.AdamW(
            self.parameters(),  # 包括encoder和classifier的所有参数
            lr=self.hparams.get('learning_rate', 1e-3),  # 从hparams获取学习率，默认1e-3
            weight_decay=self.hparams.get('weight_decay', 1e-4),  # 权重衰减，默认1e-4
            amsgrad=True  # 使用AMSGrad变体提高稳定性
        )
        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
