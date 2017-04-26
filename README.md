# Patch Maker

画像から指定されたサイズのパッチを作成する。

## Install
```
$ python setup.py Install
```

## Uninstall
```
pip uninstall patch_maker
```
or
```
python setup.py install --record files.txt
cat files.txt | xargs sudo rm -rf
```

## Usage
* `patch_maker.generate_patch`**`(image, point, size, padding='MIRROR', to_image=True)`**
  画像の指定された座標からパッチを切り出す.
  * **Parameters:**
    * **image** PIL.Image. 元となる画像.
    * **point** (x, y). パッチの中心となる座標.
    * **size** (width, height). パッチのサイズ.
    * **padding** パッチの切り出し方のオプション.`'MIRROR'`,`'SAME'`,`'VALID'`が使用できる.デフォルトは`'MIRROR'`.`'MIRROR'`ではパッチ領域が元画像からはみ出た場合、はみ出た部分をミラーリングする.`'SAME'`ではゼロパディングを行う.`'VALID'`でははみ出ないようにパッチ領域を変更する.
    * **to_image** パッチを`PIL.Image`に変換するかどうか.デフォルトは`True`.`False`の場合は`numpy.ndarray`としてデータが返ってくる.
  * **Returns:**
    * `PIL.Image`または`numpy.ndarray`オブジェクト.

* `patch_maker.generate_patches`**`(image, size, interval=1, padding='MIRROR', to_image=True)`**
  画像の座標(0, 0)から連続して指定されたサイズのパッチを切り出す.
  * **Parameters:**
    * **image** PIL.Image. 元となる画像.
    * **size** (width, height). パッチのサイズ.
    * **interval** パッチごとの間隔.
    * **padding** パッチの切り出し方のオプション.`'MIRROR'`,`'SAME'`,`'VALID'`が使用できる.デフォルトは`'MIRROR'`.`'MIRROR'`ではパッチ領域が元画像からはみ出た場合、はみ出た部分をミラーリングする.`'SAME'`ではゼロパディングを行う.`'VALID'`でははみ出ないようにパッチ領域を変更する.
    * **to_image** パッチを`PIL.Image`に変換するかどうか.デフォルトは`True`.`False`の場合は`numpy.ndarray`としてデータが返ってくる.
  * **Returns:**
    * `PIL.Image`または`numpy.ndarray`オブジェクトを返すジェネレータ.
