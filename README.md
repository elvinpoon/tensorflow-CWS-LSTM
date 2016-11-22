
# DeepSeg 1.0

### 1 在NLPCC数据集上validation accuracy 达到96.5%
### 2 模型是windowed LSTM， tagging scheme是 ‘BC‘，可以通过更改num_class变为‘BMES’.
### 3 字向量是在500M搜狗语料和本身的语料上训练得到的
## 调用命令 python model_LSTM.py --train=data/trainSeg.txt --validation=valSeg.txt --model=model --iters=50

# 新增联合模型 pos+ seg；经测试能提高分词约0.5%的F值，pos约2%的F值
