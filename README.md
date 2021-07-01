# WGAN for melenoma

## 生成新的图片

```shell
python inference.py
```

-N 可指定数量，默认1000张

生成的图片在**generated_image文件夹**内



## train

使用WGAN_train.py，在original_image/melanoma_image_meta/melanoma内放入所有的黑色素瘤图片
