import PIL.Image as image
row, col = 300, 200
pic_new = image.new("RGB", (row, col))  # "L" 灰階; "RGB" 彩色
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j),(223,250,223))
pic_new.save("test.jpg", "JPEG")
# pic_new.show() does not work in Windows
