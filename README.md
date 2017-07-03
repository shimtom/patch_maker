# Patch Maker

画像から指定されたサイズのパッチを作成する。

## Requirement
* numpy package
* pillow package

## Install
```
$ pip3 install -e .
```

## Uninstall
```
$ pip3 uninstall patch_maker
```

## Usage
### Command
```
$ makepatch --image path_to_image_dir --save path_to_save_dir --size width height
```
```
usage: makepatch [-h] [--strides STRIDES STRIDES]
                 [--bounds BOUNDS BOUNDS BOUNDS BOUNDS]
                 images [images ...] size size save

positional arguments:
  images                image. support extensions:[.png|.jpg|.tif]
  size                  patch size
  save                  save directory.

optional arguments:
  -h, --help            show this help message and exit
  --strides STRIDES STRIDES
                        strides
  --bounds BOUNDS BOUNDS BOUNDS BOUNDS
                        bounds
```

### Library
* `patch_maker.generate_patch`**`(image, point, size, padding=Padding.MIRROR, to_image=True)`**
  画像の指定された座標からパッチを切り出す.
  * **Parameters:**
    * **image** PIL.Image. 元となる画像.
    * **point** (x, y). パッチの中心となる座標.
    * **size** (width, height). パッチのサイズ.最大は元画像のサイズの2倍.最小は(1, 1).
    * **padding** パッチの切り出し方のオプション.
    `Padding.MIRROR`,`Padding.SAME`,`Padding.VALID`が使用できる.デフォルトは`Padding.MIRROR`.`Padding.MIRROR`ではパッチ領域が元画像からはみ出た場合、はみ出た部分をミラーリングする.`Padding.MIRROR`ではゼロパディングを行う.`Padding.MIRROR`でははみ出ないようにパッチ領域を変更する.
    * **to_image** パッチを`PIL.Image`に変換するかどうか.デフォルトは`True`.`False`の場合は`numpy.ndarray`としてデータが返ってくる.
  * **Returns:**
    * `PIL.Image`または`numpy.ndarray`オブジェクト.

* `patch_maker.generate_patches`**`(image, size, interval=1, padding='MIRROR', to_image=True)`**
  画像の座標(0, 0)から連続して指定されたサイズのパッチを切り出す.
  * **Parameters:**
    * **image** PIL.Image. 元となる画像.
    * **size** (width, height). パッチのサイズ.
    * **interval** パッチごとの間隔.
    * **padding** パッチの切り出し方のオプション.
    `Padding.MIRROR`,`Padding.SAME`,`Padding.VALID`が使用できる.デフォルトは`Padding.MIRROR`.`Padding.MIRROR`ではパッチ領域が元画像からはみ出た場合、はみ出た部分をミラーリングする.`Padding.MIRROR`ではゼロパディングを行う.`Padding.MIRROR`でははみ出ないようにパッチ領域を変更する.
    * **to_image** パッチを`PIL.Image`に変換するかどうか.デフォルトは`True`.`False`の場合は`numpy.ndarray`としてデータが返ってくる.
  * **Yield:**
    * `PIL.Image`または`numpy.ndarray`オブジェクト.
