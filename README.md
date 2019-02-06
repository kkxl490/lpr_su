# High Accuracy Chinese Plate Recognition Framework

**Fork原项目python实现，并作部分修改**
### 介绍
This research aims at simply developping plate recognition project based on deep learning methods, with low complexity and high speed. This 
project has been used by some commercial corporations. Free and open source, deploying by Zeusee. 


##### 更新热点:

- 添加的新的Python 序列模型-识别率大幅提高(尤其汉字)(2018.3.12)
- 新添加了HyperLPR Lite 只需要一个文件 160行代码即可完全整个车牌识别流程.
- 提供精确定位的车牌矩形框(2018.3.12)	

#### 相关资源 

+ [相关技术博客](http://blog.csdn.net/relocy/article/details/78705662)(技术文章会在接下来的几个月的时间内连续更新)。
+ [带UI界面的工程](https://pan.baidu.com/s/1cNWpK6)(感谢群内小伙伴的工作)。
+ [端到端(多标签分类)训练代码](https://github.com/LCorleone/hyperlpr-train_e2e)(感谢群内小伙伴的工作)。
+ [端到端(CTC)训练代码](https://github.com/armaab/hyperlpr-train)(感谢群内小伙伴工作)。


### TODO

+ 提供字符字符识别的训练代码
+ 改进精定位方法
+ C++版的端到端识别模型

### 特性

+ 速度快 720p ，单核 Intel 2.2G CPU (macbook Pro 2015)平均识别时间低于100ms
+ 基于端到端的车牌识别无需进行字符分割
+ 识别率高,仅仅针对车牌ROI在EasyPR数据集上，0-error达到 95.2%, 1-error识别率达到 97.4% (指在定位成功后的车牌识别率)
+ 轻量 总代码量不超1k行

### 模型资源说明

+ cascade.xml  检测模型 - 目前效果最好的cascade检测模型
+ cascade_lbp.xml  召回率效果较好，但其错检太多
+ char_chi_sim.h5 Keras模型-可识别34类数字和大写英文字  使用14W样本训练 
+ char_rec.h5 Keras模型-可识别34类数字和大写英文字  使用7W样本训练 
+ ocr_plate_all_w_rnn_2.h5 基于CNN的序列模型
+ ocr_plate_all_gru.h5 基于GRU的序列模型从OCR模型修改，效果目前最好但速度较慢，需要20ms。
+ plate_type.h5 用于车牌颜色判断的模型
+ model12.h5 左右边界回归模型


### Python 依赖

+ Keras (>2.0.0)
+ Theano(>0.9) or Tensorflow(>1.1.x)
+ Numpy (>1.10)
+ Scipy (0.19.1)
+ OpenCV(>3.0)
+ Scikit-image (0.13.0)
+ PIL


### 简单使用方式

推荐使用新更新的HyperLPR Lite，仅需一单独文件。

```python
import HyperLPRLite as pr
import cv2
import numpy as np
grr = cv2.imread("images_rec/1.jpg")
model = pr.LPR("model/cascade.xml","model/model12.h5","model/ocr_plate_all_gru.h5")

def drawRectBox(image,rect,addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText.decode("utf-8"), (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex
    
for pstr,confidence,rect in model.SimpleRecognizePlateByE2E(grr):
        if confidence>0.7:
            image = drawRectBox(grr, rect, pstr+" "+str(round(confidence,3)))
            print("plate_str",pstr)
            print("plate_confidence",confidence)


cv2.imshow("image",image)
cv2.waitKey(0)

```




### 可识别和待支持的车牌的类型

- [x] 单行蓝牌
- [x] 单行黄牌
- [x] 新能源车牌
- [x] 白色警用车牌
- [x] 使馆/港澳车牌
- [x] 教练车牌
- [x] 武警车牌
- [ ] 民航车牌
- [ ] 双层黄牌
- [ ] 双层武警
- [ ] 双层军牌
- [ ] 双层农用车牌
- [ ] 双层个性化车牌


###### Note:由于训练的时候样本存在一些不均衡的问题,一些特殊车牌存在一定识别率低下的问题，如(使馆/港澳车牌)，会在后续的版本进行改进。


### 作者和贡献者信息：
##### 作者昵称不分前后 
+ Jack Yu 作者(jack-yu-business@foxmail.com / https://github.com/szad670401)
+ lsy17096535 整理(https://github.com/lsy17096535)
+ xiaojun123456 IOS贡献(https://github.com/xiaojun123456)
+ sundyCoder Android第三方贡献(https://github.com/sundyCoder)
+ coleflowers php贡献(@coleflowers)
+ Free&Easy 资源贡献 
+ 海豚嘎嘎 LBP cascade检测器训练
